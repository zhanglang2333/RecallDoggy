from collections import deque
import subprocess
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
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
import urllib.request
from urllib.parse import quote
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()
from typing import List
from datetime import datetime, timedelta
import cnlunar
import logging
from logging.handlers import RotatingFileHandler

from memory import TZ_CN, LEVEL_ORDER
from store import ZillizMemoryStore

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

AUTH_FILE = os.path.join(os.path.dirname(__file__), ".auth")

encoder = None
store = None
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

mcp_server = FastMCP("RecallDoggy", transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
mcp_http_app = mcp_server.streamable_http_app()

async def _startup():
    global encoder, store
    logger.info("启动服务...")
    store = ZillizMemoryStore(
        uri=os.getenv("ZILLIZ_URI"),
        token=os.getenv("ZILLIZ_TOKEN")
    )
    await store.connect()
    encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    logger.info("模型加载成功")

@asynccontextmanager
async def combined_lifespan(application):
    async with mcp_http_app.router.lifespan_context(application):
        await _startup()
        yield

app = FastAPI(default_response_class=UTF8JSONResponse, lifespan=combined_lifespan)
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
    now = time.time()
    if ip not in login_attempts:
        login_attempts[ip] = deque()
    attempts = login_attempts[ip]
    while attempts and now - attempts[0] > 600:
        attempts.popleft()
    if len(attempts) >= 5:
        wait = int(600 - (now - attempts[0])) + 1
        return {"success": False, "msg": f"尝试过多，{wait // 60 + 1}分钟后再试"}
    password = data.get("password", "")
    stored_hash = get_password_hash()
    if stored_hash and bcrypt.checkpw(password.encode(), stored_hash.encode()):
        request.session["authed"] = True
        login_attempts.pop(ip, None)
        logger.info(f"登录成功 | IP:{ip}")
        return {"success": True}
    else:
        attempts.append(now)
        left = 5 - len(attempts)
        logger.warning(f"登录失败 | IP:{ip} | 窗口内:{len(attempts)}次")
        if left > 0:
            return {"success": False, "msg": f"密码错误，还剩{left}次"}
        return {"success": False, "msg": "尝试过多，10分钟后再试"}

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login")

# === 页面路由 ===
@app.get("/health")
async def health():
    cnt = await store.count() if store else 0
    return {"status": "ok", "version": "1.6.0", "entities": cnt}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    cnt = await store.count() if store else 0
    return templates.TemplateResponse("index.html", {"request": request, "total": cnt})

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
        return await store.write(
            content=req.content, embedding=embedding,
            category=req.category, tags=req.tags,
            memory_level=req.memory_level
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search_knowledge(req: SearchRequest):
    try:
        query_vec = encoder.encode(req.query).tolist()
        return await store.search(query_vec, req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def stats_api():
    try:
        return await store.stats()
    except Exception as e:
        cnt = await store.count() if store else 0
        return {"total": cnt, "collection": "ai_knowledge", "levels": {}, "trend": {}, "recent": []}

@app.delete("/api/delete/{doc_id}")
async def delete_knowledge(doc_id: str):
    try:
        await store.delete(doc_id)
        return {"status": "success", "message": "已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/list")
async def list_knowledge(limit: int = 50, offset: int = 0):
    try:
        return await store.list_all(limit, offset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/update/{doc_id}")
async def update_knowledge(doc_id: str, req: UpdateRequest):
    try:
        embedding = encoder.encode(req.content).tolist()
        result = await store.update(doc_id, req.content, embedding, req.category, req.tags)
        if result is None:
            raise HTTPException(status_code=404, detail="记忆不存在")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/set_level/{doc_id}")
async def set_level(doc_id: str, level: str):
    if level not in LEVEL_ORDER:
        raise HTTPException(status_code=400, detail=f"无效层级，可选: {LEVEL_ORDER}")
    try:
        result = await store.set_level(doc_id, level)
        if result is None:
            raise HTTPException(status_code=404, detail="记忆不存在")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cleanup")
async def cleanup(req: CleanupRequest):
    try:
        deleted = await store.cleanup(req.threshold)
        return {"message": f"已清理 {deleted} 条衰减记忆", "deleted": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export")
async def export_all():
    try:
        items = await store.export_all()
        data = {
            "exported_at": datetime.now(TZ_CN).isoformat(),
            "total": len(items), "data": items,
        }
        fname = f"kb_export_{datetime.now(TZ_CN).strftime('%Y-%m-%d_%H%M')}.json"
        body = json.dumps(data, ensure_ascii=False, indent=2)
        return Response(
            content=body.encode("utf-8"),
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={fname}"}
        )
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

@app.get("/api/dashboard")
async def dashboard_data():
    try:
        return await store.dashboard()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

# === MCP工具 ===
@mcp_server.tool()
async def mcp_search(query: str, top_k: int = 5) -> str:
    """在记忆库中语义搜索。

    query: 提取核心关键词或短语搜索，不要把用户的完整对话原文丢进来。
      好的query: "RecallDoggy部署端口" "小墨生日" "牙套品牌"
      差的query: "你之前有没有记过我的生日是哪天来着"
    top_k: 返回条数，默认5。
    返回: permanent置顶记忆(不占top_k) + 按 similarity*0.7+retention*0.3 加权排序的结果。
    每条结果含 id/content/category/tags/similarity/memory_level/retention/recall_count。
    """
    query_vec = encoder.encode(query).tolist()
    result = await store.search(query_vec, top_k)
    return json.dumps(result, ensure_ascii=False)

@mcp_server.tool()
async def mcp_write(content: str, category: str = "通用", tags: str = "", memory_level: str = "flash") -> str:
    """向记忆库写入一条记忆。

    content: 记忆正文。写完整清晰的陈述句，避免模糊指代。
    category: 分类，仅限以下值：通用 / 技术 / 生活 / 学习 / 纪念日 / 人物 / 项目。
      不要自创分类。category设为纪念日时自动升为permanent。
    tags: 逗号分隔的字符串，如 "Python,FastAPI,部署"。
      纪念日必须包含日期加类型标签，如 "08-13,solar" 或 "六月廿三,lunar"。
      不要传JSON数组，只接受逗号分隔纯文本。
    memory_level: 记忆层级:
      flash: 临时信息（默认，24h半衰期，没人搜就衰减消失）
      short: 近期有用（7天半衰期）
      long: 重要但非核心（30天半衰期）
      permanent: 绝不能忘的核心信息（永不衰减）
      拿不准就用flash，系统会根据召回次数自动升级。
    返回写入结果含id。相同内容MD5去重不会重复写入。
    """
    embedding = encoder.encode(content).tolist()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    level = memory_level
    if category == "纪念日":
        level = "permanent"
    result = await store.write(content, embedding, category, tag_list, level)
    return json.dumps(result, ensure_ascii=False)

@mcp_server.tool()
async def mcp_delete(doc_id: str) -> str:
    """删除一条记忆。

    doc_id: 记忆ID，从 mcp_search 返回结果的 id 字段获取。不要猜测或编造ID。
    """
    await store.delete(doc_id)
    return f"已删除 {doc_id}"

@mcp_server.tool()
async def mcp_stats() -> str:
    """查看记忆库统计：总数加各层级(flash/short/long/permanent)分布数量。"""
    result = await store.stats()
    return json.dumps(result, ensure_ascii=False)

@mcp_server.tool()
async def mcp_today() -> str:
    """获取今天的日期信息：公历、农历、干支、生肖、节气、节日、自定义纪念日。无需任何参数。"""
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
        res = await store.query_by_category("纪念日", ["content", "tags"], 100)
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
async def mcp_weather(city: str) -> str:
    """查询指定城市实时天气。

    city: 城市名（必填）。不要假设默认城市，不确定就问用户。
    返回: 温度、体感温度、湿度、天气描述、今日温度范围。
    """
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
app.mount("/mcp-http", mcp_http_app)

@app.post("/api/update-system")
async def update_system():
    try:
        result = subprocess.run(
            ["bash", "/root/update.sh"],
            capture_output=True, text=True, timeout=60
        )
        return {
            "status": "success" if result.returncode == 0 else "error",
            "output": result.stdout,
            "error": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "update timeout 60s"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import sys
    if "--stdio" in sys.argv:
        mcp_server.run(transport="stdio")
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8001)
