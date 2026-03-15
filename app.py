from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse, JSONResponse, Response
import json
import os
import time
import math
import bcrypt
import urllib.request
from urllib.parse import quote
from dotenv import load_dotenv
load_dotenv()
from typing import List, Optional
from datetime import datetime, timedelta, timezone
import cnlunar
import hashlib
import logging
from logging.handlers import RotatingFileHandler

# === 分层记忆常量 ===
HALF_LIFE = {"flash": 24, "short": 168, "long": 720, "permanent": None}
UPGRADE_THRESHOLDS = {1: "short", 4: "long", 10: "permanent"}
LEVEL_ORDER = ["flash", "short", "long", "permanent"]
TZ_CN = timezone(timedelta(hours=8))

def now_ms():
    return int(datetime.now(TZ_CN).timestamp() * 1000)

def calc_retention(memory_level, last_recall_ts):
    if memory_level == "permanent":
        return 1.0
    S = HALF_LIFE.get(memory_level, 24)
    t_hours = max(0, (now_ms() - last_recall_ts) / 3600000)
    return math.exp(-t_hours / S)

def check_upgrade(memory_level, recall_count):
    if memory_level == "permanent":
        return "permanent"
    for threshold in sorted(UPGRADE_THRESHOLDS.keys(), reverse=True):
        if recall_count >= threshold:
            target = UPGRADE_THRESHOLDS[threshold]
            if LEVEL_ORDER.index(target) > LEVEL_ORDER.index(memory_level):
                return target
    return memory_level

# === 日志 ===
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("recalldoggy")
logger.setLevel(logging.INFO)
_fh = RotatingFileHandler(os.path.join(LOG_DIR, "app.log"), maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
_fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(_fh)
_ch = logging.StreamHandler()
_ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(_ch)

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = "ai_knowledge"
EMBEDDING_DIM = 384
AUTH_FILE = os.path.join(os.path.dirname(__file__), ".auth")

encoder = None
collection = None
login_attempts = {}

def get_password_hash():
    if os.path.exists(AUTH_FILE):
        with open(AUTH_FILE, "r") as f:
            return f.read().strip()
    return None

def set_password_hash(password):
    h = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    with open(AUTH_FILE, "w") as f:
        f.write(h)

class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        if request.url.path in ["/favicon.ico", "/health"]:
            return await call_next(request)
        response = await call_next(request)
        duration = round((time.time() - start) * 1000, 1)
        ip = request.client.host if request.client else "unknown"
        level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(level, f"{request.method} {request.url.path} | {response.status_code} | {duration}ms | {ip}")
        return response

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        public_paths = ["/login", "/setup", "/favicon.ico", "/health"]
        if request.url.path in public_paths:
            return await call_next(request)
        if not get_password_hash():
            if not request.url.path.startswith("/mcp"):
                return RedirectResponse(url="/setup")
        if request.url.path.startswith("/mcp"):
            token = request.headers.get("Authorization", "").replace("Bearer ", "")
            if token != os.getenv("MCP_TOKEN"):
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
            return await call_next(request)
        if not request.session.get("authed"):
            if request.url.path.startswith("/api"):
                return JSONResponse({"error": "Not logged in"}, status_code=401)
            return RedirectResponse(url="/login")
        return await call_next(request)

class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"
    def render(self, content):
        return json.dumps(content, ensure_ascii=False).encode("utf-8")

app = FastAPI(default_response_class=UTF8JSONResponse)
app.add_middleware(AuthMiddleware)
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "recalldoggy-default-secret-change-me"), max_age=60*60*24*7)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RequestLogMiddleware)

templates = Jinja2Templates(directory="templates")

class WriteRequest(BaseModel):
    content: str
    category: str = "通用"
    tags: List[str] = []
    memory_level: str = "flash"

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class UpdateRequest(BaseModel):
    content: str
    category: str = "通用"
    tags: List[str] = []

class CleanupRequest(BaseModel):
    threshold: float = 0.05

@app.on_event("startup")
async def startup():
    global encoder, collection
    logger.info("启动服务...")
    connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    logger.info("已连接 Zilliz")
    encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    logger.info("模型加载成功")

    if utility.has_collection(COLLECTION_NAME):
        old = Collection(COLLECTION_NAME)
        field_names = [f.name for f in old.schema.fields]
        if "memory_level" not in field_names:
            logger.warning("旧schema检测到，正在重建collection...")
            utility.drop_collection(COLLECTION_NAME)
        else:
            collection = old
            collection.load()
            logger.info(f"知识库就绪，当前知识数：{collection.num_entities}")
            return

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="timestamp", dtype=DataType.INT64),
        FieldSchema(name="memory_level", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="recall_count", dtype=DataType.INT64),
        FieldSchema(name="last_recall", dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields=fields)
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    collection.create_index(field_name="embedding", index_params={"metric_type": "COSINE", "index_type": "AUTOINDEX", "params": {}})
    collection.load()
    logger.info("新collection已创建（含分层记忆字段）")

ALL_FIELDS = ["id", "content", "category", "tags", "timestamp", "memory_level", "recall_count", "last_recall"]

def format_item(r, similarity=None):
    ts = r.get("timestamp", 0)
    level = r.get("memory_level", "flash")
    rc = r.get("recall_count", 0)
    lr = r.get("last_recall", ts)
    ret = calc_retention(level, lr)
    item = {
        "id": r.get("id"),
        "content": r.get("content"),
        "category": r.get("category"),
        "tags": r.get("tags", "").split(","),
        "time": datetime.fromtimestamp(ts / 1000, tz=TZ_CN).strftime("%Y-%m-%d %H:%M"),
        "memory_level": level,
        "recall_count": rc,
        "retention": round(ret * 100, 2),
    }
    if similarity is not None:
        item["similarity"] = similarity
    return item

def do_recall(doc_id, current_level, current_count):
    """更新recall_count和last_recall，检查升级"""
    new_count = current_count + 1
    new_level = check_upgrade(current_level, new_count)
    new_lr = now_ms()
    # delete + reinsert 更新字段
    old = collection.query(expr=f'id == "{doc_id}"', output_fields=ALL_FIELDS, limit=1)
    if not old:
        return
    r = old[0]
    collection.delete(expr=f'id == "{doc_id}"')
    data = [
        [r["id"]], [encoder.encode(r["content"]).tolist()],
        [r["content"]], [r["category"]], [r["tags"]],
        [r["timestamp"]], [new_level], [new_count], [new_lr]
    ]
    collection.insert(data)
    if new_level != current_level:
        logger.info(f"记忆升级: {doc_id} {current_level} → {new_level} (recall={new_count})")

# === 认证路由 ===
@app.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request):
    if get_password_hash():
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("setup.html", {"request": request})

@app.post("/setup")
async def do_setup(request: Request):
    if get_password_hash():
        return {"success": False, "msg": "密码已设置"}
    data = await request.json()
    pw = data.get("password", "")
    if len(pw) < 6:
        return {"success": False, "msg": "密码至少6位"}
    set_password_hash(pw)
    request.session["authed"] = True
    return {"success": True}

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if request.session.get("authed"):
        return RedirectResponse(url="/")
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def do_login(request: Request):
    data = await request.json()
    ip = request.client.host
    if ip in login_attempts:
        count, lock_until = login_attempts[ip]
        if lock_until and time.time() < lock_until:
            remaining = int((lock_until - time.time()) / 60) + 1
            return {"success": False, "msg": f"尝试过多，{remaining}分钟后再试"}
    password = data.get("password", "")
    stored_hash = get_password_hash()
    if stored_hash and bcrypt.checkpw(password.encode(), stored_hash.encode()):
        request.session["authed"] = True
        login_attempts.pop(ip, None)
        logger.info(f"登录成功 | IP:{ip}")
        return {"success": True}
    else:
        count = login_attempts.get(ip, [0, None])[0] + 1
        lock_until = time.time() + 600 if count >= 5 else None
        login_attempts[ip] = [count, lock_until]
        logger.warning(f"登录失败 | IP:{ip} | 累计:{count}次")
        return {"success": False, "msg": "密码错误"}

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login")

# === 页面路由 ===
@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.5.0", "entities": collection.num_entities if collection else 0}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "total": collection.num_entities if collection else 0})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    return templates.TemplateResponse("logs.html", {"request": request})

# === 核心API ===
@app.post("/api/write")
async def write_knowledge(req: WriteRequest):
    try:
        embedding = encoder.encode(req.content).tolist()
        doc_id = hashlib.md5(req.content.encode()).hexdigest()
        result = collection.query(expr=f'id == "{doc_id}"', output_fields=["id"], limit=1)
        if result:
            return {"status": "exists", "message": "知识已存在"}
        level = req.memory_level if req.memory_level in LEVEL_ORDER else "flash"
        ts = now_ms()
        data = [
            [doc_id], [embedding], [req.content], [req.category],
            [",".join(req.tags)], [ts], [level], [0], [ts]
        ]
        collection.insert(data)
        collection.flush()
        logger.info(f"写入[{level}]: {req.content[:50]} | {req.category}")
        return {"status": "success", "message": "写入成功", "id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search_knowledge(req: SearchRequest):
    try:
        query_vec = encoder.encode(req.query).tolist()

        # 1. 先捞permanent记忆（始终返回，不占top_k）
        perm_results = collection.query(
            expr='memory_level == "permanent"',
            output_fields=ALL_FIELDS, limit=100
        )
        permanent = [format_item(r) for r in perm_results]

        # 2. 语义搜索
        results = collection.search(
            data=[query_vec], anns_field="embedding",
            param={"metric_type": "COSINE"}, limit=req.top_k + len(perm_results),
            output_fields=["content", "category", "tags", "timestamp", "memory_level", "recall_count", "last_recall"]
        )

        perm_ids = {r.get("id") for r in perm_results}
        items = []
        for hits in results:
            for hit in hits:
                if hit.id in perm_ids:
                    continue
                level = hit.entity.get("memory_level", "flash")
                lr = hit.entity.get("last_recall", hit.entity.get("timestamp", 0))
                ret = calc_retention(level, lr)
                sim = hit.score
                final = sim * 0.7 + ret * 0.3
                items.append({
                    "hit": hit,
                    "final_score": final,
                    "retention": ret,
                    "similarity": sim
                })

        items.sort(key=lambda x: x["final_score"], reverse=True)
        items = items[:req.top_k]

        # 3. 更新recall_count（异步不阻塞响应太复杂，同步做）
        for it in items:
            h = it["hit"]
            try:
                do_recall(h.id, h.entity.get("memory_level", "flash"), h.entity.get("recall_count", 0))
            except Exception:
                pass
        for r in perm_results:
            try:
                do_recall(r["id"], "permanent", r.get("recall_count", 0))
            except Exception:
                pass

        output = []
        for it in items:
            h = it["hit"]
            output.append({
                "id": h.id,
                "content": h.entity.get("content"),
                "category": h.entity.get("category"),
                "tags": h.entity.get("tags", "").split(","),
                "similarity": round(it["similarity"] * 100, 2),
                "time": datetime.fromtimestamp(h.entity.get("timestamp", 0) / 1000, tz=TZ_CN).strftime("%Y-%m-%d %H:%M"),
                "memory_level": h.entity.get("memory_level", "flash"),
                "recall_count": h.entity.get("recall_count", 0) + 1,
                "retention": round(it["retention"] * 100, 2),
            })

        logger.info(f"搜索: {req.query} | top_k:{req.top_k} | 结果:{len(output)}条 | permanent:{len(permanent)}条")
        return {"results": output, "permanent": permanent}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def stats_api():
    try:
        all_data = collection.query(
            expr='id != ""',
            output_fields=["memory_level", "content", "category", "timestamp"],
            limit=16384
        )
        levels = {"flash": 0, "short": 0, "long": 0, "permanent": 0}
        for r in all_data:
            lv = r.get("memory_level", "flash")
            levels[lv] = levels.get(lv, 0) + 1
        from datetime import datetime, timedelta
        now = datetime.now()
        trend = {}
        for i in range(6, -1, -1):
            day = now - timedelta(days=i)
            trend[day.strftime("%m-%d")] = 0
        for r in all_data:
            ts = r.get("timestamp", 0)
            if ts:
                dt = datetime.fromtimestamp(ts / 1000)
                key = dt.strftime("%m-%d")
                if key in trend:
                    trend[key] += 1
        sorted_data = sorted(all_data, key=lambda x: x.get("timestamp", 0), reverse=True)[:10]
        recent = []
        for r in sorted_data:
            ts = r.get("timestamp", 0)
            time_str = ""
            if ts:
                time_str = datetime.fromtimestamp(ts / 1000).strftime("%m-%d %H:%M")
            recent.append({"category": r.get("category", ""), "content": r.get("content", "")[:80], "time": time_str})
        return {"total": collection.num_entities, "collection": COLLECTION_NAME, "levels": levels, "trend": trend, "recent": recent}
    except Exception as e:
        return {"total": collection.num_entities, "collection": COLLECTION_NAME, "levels": {}, "trend": {}, "recent": []}
@app.delete("/api/delete/{doc_id}")
async def delete_knowledge(doc_id: str):
    try:
        collection.delete(expr=f'id == "{doc_id}"')
        logger.warning(f"删除: {doc_id}")
        return {"status": "success", "message": "已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/list")
async def list_knowledge(limit: int = 50, offset: int = 0):
    try:
        results = collection.query(
            expr='id != ""', output_fields=ALL_FIELDS,
            limit=limit, offset=offset
        )
        items = [format_item(r) for r in results]
        return {"results": items, "total": collection.num_entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/update/{doc_id}")
async def update_knowledge(doc_id: str, req: UpdateRequest):
    try:
        old = collection.query(expr=f'id == "{doc_id}"', output_fields=ALL_FIELDS, limit=1)
        if not old:
            raise HTTPException(status_code=404, detail="记忆不存在")
        r = old[0]
        collection.delete(expr=f'id == "{doc_id}"')
        vector = encoder.encode(req.content).tolist()
        data = [
            [doc_id], [vector], [req.content], [req.category],
            [",".join(req.tags)], [r.get("timestamp", now_ms())],
            [r.get("memory_level", "flash")], [r.get("recall_count", 0)], [r.get("last_recall", now_ms())]
        ]
        collection.insert(data)
        collection.flush()
        logger.info(f"更新: {doc_id}")
        return {"message": "更新成功", "id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/set_level/{doc_id}")
async def set_level(doc_id: str, level: str):
    if level not in LEVEL_ORDER:
        raise HTTPException(status_code=400, detail=f"无效层级，可选: {LEVEL_ORDER}")
    try:
        old = collection.query(expr=f'id == "{doc_id}"', output_fields=ALL_FIELDS, limit=1)
        if not old:
            raise HTTPException(status_code=404, detail="记忆不存在")
        r = old[0]
        collection.delete(expr=f'id == "{doc_id}"')
        data = [
            [r["id"]], [encoder.encode(r["content"]).tolist()],
            [r["content"]], [r["category"]], [r["tags"]],
            [r["timestamp"]], [level], [r.get("recall_count", 0)], [r.get("last_recall", now_ms())]
        ]
        collection.insert(data)
        collection.flush()
        logger.info(f"层级变更: {doc_id} → {level}")
        return {"message": f"已设为 {level}", "id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cleanup")
async def cleanup(req: CleanupRequest):
    try:
        all_data = collection.query(expr='memory_level != "permanent"', output_fields=ALL_FIELDS, limit=16384)
        to_delete = []
        for r in all_data:
            ret = calc_retention(r.get("memory_level", "flash"), r.get("last_recall", r.get("timestamp", 0)))
            if ret < req.threshold:
                to_delete.append(r["id"])
        for doc_id in to_delete:
            collection.delete(expr=f'id == "{doc_id}"')
        logger.info(f"清理完成: 删除{len(to_delete)}条，阈值{req.threshold}")
        return {"message": f"已清理 {len(to_delete)} 条衰减记忆", "deleted": len(to_delete)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export")
async def export_all():
    try:
        all_data = collection.query(expr='id != ""', output_fields=ALL_FIELDS, limit=16384)
        items = [format_item(r) for r in all_data]
        data = {"exported_at": datetime.now(TZ_CN).isoformat(), "total": len(items), "data": items}
        fname = f"kb_export_{datetime.now(TZ_CN).strftime('%Y-%m-%d_%H%M')}.json"
        body = json.dumps(data, ensure_ascii=False, indent=2)
        return Response(content=body.encode("utf-8"), media_type="application/json; charset=utf-8",
                        headers={"Content-Disposition": f"attachment; filename={fname}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs")
async def view_logs(lines: int = 100):
    log_file = os.path.join(LOG_DIR, "app.log")
    if not os.path.exists(log_file):
        return {"logs": []}
    with open(log_file, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    return {"logs": all_lines[-lines:]}

# === 天气 ===
@app.get("/api/weather")
async def weather_api(city: str = "天津"):
    try:
        url = f"https://wttr.in/{quote(city)}?format=j1&lang=zh&m"
        req = urllib.request.Request(url, headers={"User-Agent": "RecallDoggy/1.5"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        current = data.get("current_condition", [{}])[0]
        forecast = data.get("weather", [])
        result = {
            "city": city,
            "temp_C": current.get("temp_C"),
            "feels_like": current.get("FeelsLikeC"),
            "humidity": current.get("humidity"),
            "desc": current.get("lang_zh", [{}])[0].get("value", current.get("weatherDesc", [{}])[0].get("value", "")),
            "wind_speed": current.get("windspeedKmph"),
            "wind_dir": current.get("winddir16Point"),
        }
        if forecast:
            today = forecast[0]
            result["max_temp"] = today.get("maxtempC")
            result["min_temp"] = today.get("mintempC")
        return result
    except Exception as e:
        return {"city": city, "error": str(e)}

# === 日历 ===
def _parse_lunar_festivals(a):
    lunar_legal = a.get_legalHolidays()
    lunar_other = a.get_otherHolidays()
    if isinstance(lunar_legal, str):
        lunar_legal = [lunar_legal] if lunar_legal else []
    if isinstance(lunar_other, str):
        lunar_other = [lunar_other] if lunar_other else []
    return list(lunar_legal) + list(lunar_other)

@app.get("/api/today")
async def api_today():
    now = datetime.now(TZ_CN).replace(tzinfo=None)
    a = cnlunar.Lunar(now, godType='8char')
    lunar_date = f"{a.lunarMonthCn}{a.lunarDayCn}"
    ganzhi_year = a.year8Char
    zodiac = a.chineseYearZodiac
    weekdays = ["星期一","星期二","星期三","星期四","星期五","星期六","星期日"]
    weekday = weekdays[now.weekday()]
    solar_term = a.todaySolarTerms
    if solar_term == "无":
        solar_term = None
    festivals = []
    solar_festivals = {
        "01-01":"元旦","02-14":"情人节","03-08":"妇女节",
        "04-01":"愚人节","05-01":"劳动节","05-04":"青年节",
        "06-01":"儿童节","09-10":"教师节","10-01":"国庆节",
        "12-24":"平安夜","12-25":"圣诞节"
    }
    solar_key = now.strftime("%m-%d")
    if solar_key in solar_festivals:
        festivals.append(solar_festivals[solar_key])
    lunar_festivals = _parse_lunar_festivals(a)
    festivals.extend([f for f in lunar_festivals if f])
    tomorrow = now + timedelta(days=1)
    a_tomorrow = cnlunar.Lunar(tomorrow, godType='8char')
    if a_tomorrow.lunarMonthCn == "正月" and a_tomorrow.lunarDayCn == "初一":
        if "除夕" not in festivals:
            festivals.append("除夕")
    return {
        "solar": now.strftime("%Y年%m月%d日"),
        "weekday": weekday,
        "lunar": f"{ganzhi_year}年（{zodiac}年）{lunar_date}",
        "solar_term": solar_term,
        "festivals": list(set(festivals))
    }

# === Dashboard API ===
@app.get("/api/dashboard")
async def dashboard_data():
    try:
        all_data = collection.query(expr='id != ""', output_fields=ALL_FIELDS, limit=16384)
        total = len(all_data)
        levels = {"flash": 0, "short": 0, "long": 0, "permanent": 0}
        cat_count = {}
        trend = {}
        now = datetime.now(TZ_CN)
        for i in range(6, -1, -1):
            day = (now - timedelta(days=i)).strftime("%m-%d")
            trend[day] = 0
        for item in all_data:
            lv = item.get("memory_level", "flash")
            levels[lv] = levels.get(lv, 0) + 1
            cat = item.get("category", "未分类")
            cat_count[cat] = cat_count.get(cat, 0) + 1
            ts = item.get("timestamp", 0)
            if ts:
                day = datetime.fromtimestamp(ts / 1000, tz=TZ_CN).strftime("%m-%d")
                if day in trend:
                    trend[day] += 1
        sorted_data = sorted(all_data, key=lambda x: x.get("timestamp", 0), reverse=True)
        recent = []
        for r in sorted_data[:10]:
            recent.append({
                "id": r.get("id"),
                "content": r.get("content", "")[:80],
                "category": r.get("category"),
                "memory_level": r.get("memory_level", "flash"),
                "time": datetime.fromtimestamp(r.get("timestamp", 0) / 1000, tz=TZ_CN).strftime("%m-%d %H:%M")
            })
        return {"total": total, "levels": levels, "categories": cat_count, "trend": trend, "recent": recent}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === MCP ===
mcp_server = FastMCP("RecallDoggy", transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))

@mcp_server.tool()
async def mcp_search(query: str, top_k: int = 5) -> str:
    """搜索记忆库。返回语义最相似的结果+permanent置顶记忆。结果含similarity、memory_level、retention等。"""
    query_vec = encoder.encode(query).tolist()
    perm_results = collection.query(expr='memory_level == "permanent"', output_fields=ALL_FIELDS, limit=100)
    results = collection.search(
        data=[query_vec], anns_field="embedding",
        param={"metric_type": "COSINE"}, limit=top_k,
        output_fields=["content", "category", "tags", "timestamp", "memory_level", "recall_count", "last_recall"]
    )
    perm_ids = {r.get("id") for r in perm_results}
    items = []
    for r in perm_results:
        items.append({"content": r["content"], "category": r["category"], "tags": r.get("tags",""), "memory_level": "permanent", "pinned": True})
    for hits in results:
        for hit in hits:
            if hit.id in perm_ids:
                continue
            level = hit.entity.get("memory_level", "flash")
            lr = hit.entity.get("last_recall", hit.entity.get("timestamp", 0))
            ret = calc_retention(level, lr)
            items.append({
                "content": hit.entity.get("content"),
                "category": hit.entity.get("category"),
                "tags": hit.entity.get("tags", ""),
                "similarity": round(hit.score * 100, 2),
                "memory_level": level,
                "retention": round(ret * 100, 2),
            })
    return json.dumps(items, ensure_ascii=False)

@mcp_server.tool()
async def mcp_write(content: str, category: str = "通用", tags: str = "", memory_level: str = "flash") -> str:
    """向记忆库写入。memory_level可选flash/short/long/permanent，纪念日自动permanent。"""
    embedding = encoder.encode(content).tolist()
    doc_id = hashlib.md5(content.encode()).hexdigest()
    result = collection.query(expr=f'id == "{doc_id}"', output_fields=["id"], limit=1)
    if result:
        return "记忆已存在"
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    level = memory_level if memory_level in LEVEL_ORDER else "flash"
    if category == "纪念日":
        level = "permanent"
    ts = now_ms()
    data = [[doc_id], [embedding], [content], [category], [",".join(tag_list)], [ts], [level], [0], [ts]]
    collection.insert(data)
    collection.flush()
    return f"写入成功[{level}]，ID: {doc_id}"

@mcp_server.tool()
async def mcp_delete(doc_id: str) -> str:
    """删除指定ID的记忆"""
    collection.delete(expr=f'id == "{doc_id}"')
    return f"已删除 {doc_id}"

@mcp_server.tool()
async def mcp_stats() -> str:
    """查看记忆库统计，含各层级数量"""
    all_data = collection.query(expr='id != ""', output_fields=["memory_level"], limit=16384)
    levels = {"flash": 0, "short": 0, "long": 0, "permanent": 0}
    for r in all_data:
        lv = r.get("memory_level", "flash")
        levels[lv] = levels.get(lv, 0) + 1
    return json.dumps({"total": collection.num_entities, "levels": levels}, ensure_ascii=False)

@mcp_server.tool()
async def mcp_today() -> str:
    """获取今天的完整日期信息（公历/农历/节气/节日/纪念日）"""
    now = datetime.now(TZ_CN).replace(tzinfo=None)
    a = cnlunar.Lunar(now, godType='8char')
    lunar_date = f"{a.lunarMonthCn}{a.lunarDayCn}"
    ganzhi_year = a.year8Char
    zodiac = a.chineseYearZodiac
    weekdays = ["星期一","星期二","星期三","星期四","星期五","星期六","星期日"]
    weekday = weekdays[now.weekday()]
    solar_term = a.todaySolarTerms
    if solar_term == "无":
        solar_term = None
    festivals = []
    solar_festivals = {"01-01":"元旦","02-14":"情人节","03-08":"妇女节","04-01":"愚人节","05-01":"劳动节","05-04":"青年节","06-01":"儿童节","09-10":"教师节","10-01":"国庆节","12-24":"平安夜","12-25":"圣诞节"}
    solar_key = now.strftime("%m-%d")
    if solar_key in solar_festivals:
        festivals.append(solar_festivals[solar_key])
    lunar_festivals = _parse_lunar_festivals(a)
    festivals.extend([f for f in lunar_festivals if f])
    custom = []
    try:
        res = collection.query(expr='category == "纪念日"', output_fields=["content", "tags"], limit=100)
        for item in res:
            tags = item.get("tags", "").split(",")
            for tag in tags:
                tag = tag.strip()
                if tag == solar_key or tag in lunar_date:
                    custom.append(item["content"])
    except Exception:
        pass
    lines_out = [f"📅 {now.strftime('%Y年%m月%d日')} {weekday}"]
    lines_out.append(f"🏮 农历：{ganzhi_year}年（{zodiac}年）{lunar_date}")
    if solar_term:
        lines_out.append(f"🌿 节气：{solar_term}")
    if festivals:
        lines_out.append(f"🎉 节日：{' / '.join(set(festivals))}")
    if custom:
        lines_out.append(f"💝 纪念日：{' / '.join(custom)}")
    return "\n".join(lines_out)

@mcp_server.tool()
async def mcp_weather(city: str = "天津") -> str:
    """查询城市天气。默认天津。返回温度、体感、湿度、风向、描述等。"""
    try:
        url = f"https://wttr.in/{quote(city)}?format=j1&lang=zh&m"
        req = urllib.request.Request(url, headers={"User-Agent": "RecallDoggy/1.5"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        current = data.get("current_condition", [{}])[0]
        forecast = data.get("weather", [])
        desc = current.get("lang_zh", [{}])[0].get("value", "")
        result = f"🌡️ {city}: {current.get('temp_C')}°C（体感{current.get('FeelsLikeC')}°C）| {desc} | 湿度{current.get('humidity')}%"
        if forecast:
            today = forecast[0]
            result += f" | 今日{today.get('mintempC')}~{today.get('maxtempC')}°C"
        return result
    except Exception as e:
        return f"天气获取失败: {e}"

app.mount("/mcp", mcp_server.sse_app())

if __name__ == "__main__":
    import sys
    if "--stdio" in sys.argv:
        mcp_server.run(transport="stdio")
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8001)
