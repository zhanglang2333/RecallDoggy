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
import bcrypt
from dotenv import load_dotenv
load_dotenv()
from typing import List
from datetime import datetime, timedelta, timezone
import cnlunar
import hashlib
import logging
from logging.handlers import RotatingFileHandler

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
login_attempts: dict = {}

def get_password_hash():
    if os.path.exists(AUTH_FILE):
        with open(AUTH_FILE, "r") as f:
            return f.read().strip()
    return None

def set_password_hash(password: str):
    h = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    with open(AUTH_FILE, "w") as f:
        f.write(h)

class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        if request.url.path in ["/favicon.ico", "/health"]:
            return await call_next(request)
        response = await call_next(request)
        duration = round((time.time() - start) * 1000, 1)
        ip = request.client.host if request.client else "unknown"
        status = response.status_code
        level = logging.WARNING if status >= 400 else logging.INFO
        logger.log(level, f"{request.method} {request.url.path} | {status} | {duration}ms | {ip}")
        return response

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
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
                return JSONResponse({"error": "未登录"}, status_code=401)
            return RedirectResponse(url="/login")
        return await call_next(request)

class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"
    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False).encode("utf-8")

app = FastAPI(default_response_class=UTF8JSONResponse)

app.add_middleware(AuthMiddleware)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "recalldoggy-default-secret-change-me"),
    max_age=60 * 60 * 24 * 7,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLogMiddleware)

templates = Jinja2Templates(directory="templates")

class WriteRequest(BaseModel):
    content: str
    category: str = "通用"
    tags: List[str] = []

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class UpdateRequest(BaseModel):
    content: str
    category: str = "通用"
    tags: List[str] = []

@app.on_event("startup")
async def startup():
    global encoder, collection
    logger.info("启动服务...")
    connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    logger.info("已连接 Zilliz")
    encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    logger.info("模型加载成功")
    if utility.has_collection(COLLECTION_NAME):
        collection = Collection(COLLECTION_NAME)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields=fields)
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        index_params = {"metric_type": "COSINE", "index_type": "AUTOINDEX", "params": {}}
        collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()
    logger.info(f"知识库就绪，当前知识数：{collection.num_entities}")

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

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "1.4.0",
        "entities": collection.num_entities if collection else 0
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "total": collection.num_entities if collection else 0
    })

@app.post("/api/write")
async def write_knowledge(req: WriteRequest):
    try:
        embedding = encoder.encode(req.content).tolist()
        doc_id = hashlib.md5(req.content.encode()).hexdigest()
        result = collection.query(expr=f'id == "{doc_id}"', output_fields=["id"], limit=1)
        if result:
            return {"status": "exists", "message": "知识已存在"}
        data = [
            [doc_id], [embedding], [req.content], [req.category],
            [",".join(req.tags)], [int(datetime.now().timestamp() * 1000)]
        ]
        collection.insert(data)
        collection.flush()
        logger.info(f"写入: {req.content[:50]} | {req.category}")
        return {"status": "success", "message": "写入成功", "id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search_knowledge(req: SearchRequest):
    try:
        query_vec = encoder.encode(req.query).tolist()
        results = collection.search(
            data=[query_vec], anns_field="embedding",
            param={"metric_type": "COSINE"}, limit=req.top_k,
            output_fields=["content", "category", "tags", "timestamp"]
        )
        items = []
        for hits in results:
            for hit in hits:
                items.append({
                    "id": hit.id,
                    "content": hit.entity.get("content"),
                    "category": hit.entity.get("category"),
                    "tags": hit.entity.get("tags", "").split(","),
                    "similarity": round(hit.score * 100, 2),
                    "time": datetime.fromtimestamp(hit.entity.get("timestamp", 0) / 1000).strftime("%Y-%m-%d %H:%M")
                })
        logger.info(f"搜索: {req.query} | top_k:{req.top_k} | 结果:{len(items)}条")
        return {"results": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def stats_api():
    return {"total": collection.num_entities, "collection": COLLECTION_NAME}

@app.delete("/api/delete/{doc_id}")
async def delete_knowledge(doc_id: str):
    try:
        collection.delete(expr=f'id == "{doc_id}"')
        logger.warning(f"删除: {doc_id}")
        return {"status": "success", "message": "已删除"}
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

@app.get("/api/list")
async def list_knowledge(limit: int = 50, offset: int = 0):
    try:
        results = collection.query(
            expr='id != ""',
            output_fields=["id", "content", "category", "tags", "timestamp"],
            limit=limit, offset=offset
        )
        items = []
        for r in results:
            items.append({
                "id": r.get("id"),
                "content": r.get("content"),
                "category": r.get("category"),
                "tags": r.get("tags", "").split(","),
                "time": datetime.fromtimestamp(r.get("timestamp", 0) / 1000).strftime("%Y-%m-%d %H:%M")
            })
        return {"results": items, "total": collection.num_entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export")
async def export_all():
    try:
        all_data = collection.query(
            expr='id != ""',
            output_fields=["id", "content", "category", "tags", "timestamp"],
            limit=16384
        )
        items = []
        for r in all_data:
            items.append({
                "id": r.get("id"),
                "content": r.get("content"),
                "category": r.get("category"),
                "tags": r.get("tags", "").split(","),
                "timestamp": r.get("timestamp", 0)
            })
        data = {
            "exported_at": datetime.now().isoformat(),
            "total": len(items),
            "data": items
        }
        fmt = "%Y-%m-%d_%H%M"
        fname = f"kb_export_{datetime.now().strftime(fmt)}.json"
        body = json.dumps(data, ensure_ascii=False, indent=2)
        return Response(
            content=body.encode("utf-8"),
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={fname}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/update/{doc_id}")
async def update_knowledge(doc_id: str, req: UpdateRequest):
    try:
        collection.delete(expr=f'id == "{doc_id}"')
        vector = encoder.encode(req.content).tolist()
        data = [
            [doc_id], [vector], [req.content], [req.category],
            [",".join(req.tags)], [int(datetime.now().timestamp() * 1000)]
        ]
        collection.insert(data)
        collection.flush()
        logger.info(f"更新: {doc_id}")
        return {"message": "更新成功", "id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    tz_cn = timezone(timedelta(hours=8))
    now = datetime.now(tz_cn).replace(tzinfo=None)
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

mcp_server = FastMCP(
    "RecallDoggy",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False
    )
)

@mcp_server.tool()
async def mcp_search(query: str, top_k: int = 5) -> str:
    """搜索向量知识库，返回最相似的知识条目"""
    query_vec = encoder.encode(query).tolist()
    results = collection.search(
        data=[query_vec], anns_field="embedding",
        param={"metric_type": "COSINE"}, limit=top_k,
        output_fields=["content", "category", "tags", "timestamp"]
    )
    items = []
    for hits in results:
        for hit in hits:
            items.append({
                "content": hit.entity.get("content"),
                "category": hit.entity.get("category"),
                "tags": hit.entity.get("tags", ""),
                "similarity": round(hit.score * 100, 2)
            })
    return json.dumps(items, ensure_ascii=False)

@mcp_server.tool()
async def mcp_write(content: str, category: str = "通用", tags: str = "") -> str:
    """向知识库写入一条新知识"""
    embedding = encoder.encode(content).tolist()
    doc_id = hashlib.md5(content.encode()).hexdigest()
    result = collection.query(expr=f'id == "{doc_id}"', output_fields=["id"], limit=1)
    if result:
        return "知识已存在"
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    data = [
        [doc_id], [embedding], [content], [category],
        [",".join(tag_list)], [int(datetime.now().timestamp() * 1000)]
    ]
    collection.insert(data)
    collection.flush()
    return f"写入成功，ID: {doc_id}"

@mcp_server.tool()
async def mcp_delete(doc_id: str) -> str:
    """从知识库删除指定ID的知识"""
    collection.delete(expr=f'id == "{doc_id}"')
    return f"已删除 {doc_id}"

@mcp_server.tool()
async def mcp_today() -> str:
    """获取今天的完整日期信息，包括公历、农历、节气、节日、纪念日。AI应在需要感知当前日期时主动调用此工具。"""
    tz_cn = timezone(timedelta(hours=8))
    now = datetime.now(tz_cn).replace(tzinfo=None)
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
    custom = []
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
        res = client.query(
            collection_name="knowledge_base",
            filter='category == "纪念日"',
            output_fields=["content", "tags"],
            limit=100
        )
        for item in res:
            tags = item.get("tags", [])
            for tag in tags:
                tag = tag.strip()
                if tag == solar_key:
                    custom.append(item["content"])
                if tag in lunar_date:
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
async def mcp_stats() -> str:
    """查看知识库统计信息"""
    return json.dumps({"total": collection.num_entities, "collection": COLLECTION_NAME})

app.mount("/mcp", mcp_server.sse_app())

if __name__ == "__main__":
    import sys
    if "--stdio" in sys.argv:
        mcp_server.run(transport="stdio")
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8001)
