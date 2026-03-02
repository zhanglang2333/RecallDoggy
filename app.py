from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
import json
import os
import sqlite3
import math
from dotenv import load_dotenv
load_dotenv()
from typing import List, Optional
from datetime import datetime, timedelta, timezone
import cnlunar
import hashlib
import httpx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = "ai_knowledge"
EMBEDDING_DIM = 384
QWEATHER_KEY = os.getenv("QWEATHER_KEY", "")
DB_PATH = "/root/metadata.db"

encoder = None
collection = None

# ===================== SQLite =====================

STRENGTH = {
    "flash": 24,
    "short": 168,
    "long": 720,
    "permanent": float("inf")
}

def init_sqlite():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS memory_meta (
        id TEXT PRIMARY KEY,
        recall_count INTEGER DEFAULT 0,
        last_recall_time REAL,
        memory_level TEXT DEFAULT 'flash',
        created_at REAL
    )""")
    conn.commit()
    conn.close()

def get_db():
    return sqlite3.connect(DB_PATH)

def calc_retention(level, last_recall_time):
    if level == "permanent":
        return 1.0
    now = datetime.now().timestamp()
    hours = (now - last_recall_time) / 3600
    S = STRENGTH.get(level, 24)
    if S == float("inf"):
        return 1.0
    return math.exp(-hours / S)

def auto_upgrade(recall_count, current_level):
    if current_level == "permanent":
        return "permanent"
    if recall_count >= 10:
        return "permanent"
    if recall_count >= 4:
        return "long"
    if recall_count >= 1:
        return "short"
    return "flash"

def ensure_meta(db, doc_id):
    c = db.cursor()
    c.execute("SELECT id FROM memory_meta WHERE id = ?", (doc_id,))
    if not c.fetchone():
        now = datetime.now().timestamp()
        c.execute("INSERT INTO memory_meta (id, recall_count, last_recall_time, memory_level, created_at) VALUES (?, 0, ?, 'flash', ?)", (doc_id, now, now))
        db.commit()

# ===================== Models =====================

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

# ===================== Startup =====================

@app.on_event("startup")
async def startup():
    global encoder, collection
    print("启动服务...")
    init_sqlite()
    print("SQLite 就绪")
    connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    print("已连接 Zilliz")
    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("模型加载成功")
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
    # 同步已有Milvus数据到SQLite
    db = get_db()
    try:
        existing = collection.query(expr='id != ""', output_fields=["id"], limit=1000)
        for item in existing:
            ensure_meta(db, item["id"])
    except Exception:
        pass
    db.close()
    print(f"知识库就绪，当前知识数：{collection.num_entities}")

# ===================== 前端 =====================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "total": collection.num_entities if collection else 0
    })

# ===================== 写入 =====================

@app.post("/api/write")
async def write_knowledge(req: WriteRequest):
    try:
        embedding = encoder.encode(req.content).tolist()
        doc_id = hashlib.md5(req.content.encode()).hexdigest()
        result = collection.query(expr=f'id == "{doc_id}"', output_fields=["id"], limit=1)
        if result:
            return {"status": "exists", "message": "知识已存在"}
        # 自动加时间标签
        tz_cn = timezone(timedelta(hours=8))
        time_tag = datetime.now(tz_cn).strftime("%Y-%m-%d %H:%M")
        all_tags = list(req.tags) + [time_tag]
        data = [
            [doc_id], [embedding], [req.content], [req.category],
            [",".join(all_tags)], [int(datetime.now().timestamp() * 1000)]
        ]
        collection.insert(data)
        collection.flush()
        # 写SQLite
        now = datetime.now().timestamp()
        level = req.memory_level if req.memory_level in STRENGTH else "flash"
        db = get_db()
        db.execute("INSERT OR REPLACE INTO memory_meta (id, recall_count, last_recall_time, memory_level, created_at) VALUES (?, 0, ?, ?, ?)", (doc_id, now, level, now))
        db.commit()
        db.close()
        return {"status": "success", "message": "写入成功", "id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===================== 搜索 =====================

@app.post("/api/search")
async def search_knowledge(req: SearchRequest):
    try:
        db = get_db()
        c = db.cursor()
        # 1. 查所有permanent
        c.execute("SELECT id FROM memory_meta WHERE memory_level = 'permanent'")
        perm_ids = set(row[0] for row in c.fetchall())
        # 2. 拿permanent内容
        perm_items = []
        if perm_ids:
            id_list = '["' + '","'.join(perm_ids) + '"]'
            perm_results = collection.query(
                expr=f"id in {id_list}",
                output_fields=["id", "content", "category", "tags", "timestamp"]
            )
            for r in perm_results:
                c.execute("SELECT recall_count FROM memory_meta WHERE id = ?", (r["id"],))
                row = c.fetchone()
                perm_items.append({
                    "id": r["id"],
                    "content": r["content"],
                    "category": r["category"],
                    "tags": r.get("tags", "").split(","),
                    "similarity": 100.0,
                    "time": datetime.fromtimestamp(r.get("timestamp", 0) / 1000).strftime("%Y-%m-%d %H:%M"),
                    "memory_level": "permanent",
                    "recall_count": row[0] if row else 0,
                    "retention": 100.0
                })
        # 3. 向量搜索
        query_vec = encoder.encode(req.query).tolist()
        results = collection.search(
            data=[query_vec], anns_field="embedding",
            param={"metric_type": "COSINE"}, limit=req.top_k * 2,
            output_fields=["content", "category", "tags", "timestamp"]
        )
        # 4. 加权排序非permanent
        items = []
        for hits in results:
            for hit in hits:
                if hit.id in perm_ids:
                    continue
                c.execute("SELECT recall_count, last_recall_time, memory_level FROM memory_meta WHERE id = ?", (hit.id,))
                meta = c.fetchone()
                if meta:
                    recall_count, last_recall_time, level = meta
                    R = calc_retention(level, last_recall_time)
                else:
                    recall_count, level, R = 0, "flash", 1.0
                    last_recall_time = datetime.now().timestamp()
                    ensure_meta(db, hit.id)
                final_score = hit.score * 0.7 + R * 0.3
                new_level = auto_upgrade(recall_count, level)
                items.append({
                    "id": hit.id,
                    "content": hit.entity.get("content"),
                    "category": hit.entity.get("category"),
                    "tags": hit.entity.get("tags", "").split(","),
                    "similarity": round(hit.score * 100, 2),
                    "time": datetime.fromtimestamp(hit.entity.get("timestamp", 0) / 1000).strftime("%Y-%m-%d %H:%M"),
                    "memory_level": new_level,
                    "recall_count": recall_count,
                    "retention": round(R * 100, 2),
                    "_final_score": final_score
                })
        items.sort(key=lambda x: x["_final_score"], reverse=True)
        items = items[:req.top_k]
        # 5. 更新recall
        now_ts = datetime.now().timestamp()
        for item in perm_items + items:
            rid = item["id"]
            c.execute("SELECT recall_count, memory_level FROM memory_meta WHERE id = ?", (rid,))
            row = c.fetchone()
            if row:
                new_count = row[0] + 1
                new_level = auto_upgrade(new_count, row[1])
                c.execute("UPDATE memory_meta SET recall_count = ?, last_recall_time = ?, memory_level = ? WHERE id = ?", (new_count, now_ts, new_level, rid))
                item["recall_count"] = new_count
                item["memory_level"] = new_level
        db.commit()
        db.close()
        for item in items:
            item.pop("_final_score", None)
        return {"permanent": perm_items, "results": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===================== 统计 =====================

@app.get("/api/stats")
async def stats_api():
    db = get_db()
    c = db.cursor()
    counts = {}
    for level in ["flash", "short", "long", "permanent"]:
        c.execute("SELECT COUNT(*) FROM memory_meta WHERE memory_level = ?", (level,))
        counts[level] = c.fetchone()[0]
    db.close()
    return {"total": collection.num_entities, "collection": COLLECTION_NAME, "levels": counts}

# ===================== 删除 =====================

@app.delete("/api/delete/{doc_id}")
async def delete_knowledge(doc_id: str):
    try:
        collection.delete(expr=f'id == "{doc_id}"')
        db = get_db()
        db.execute("DELETE FROM memory_meta WHERE id = ?", (doc_id,))
        db.commit()
        db.close()
        return {"status": "success", "message": "已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===================== 列表 =====================

@app.get("/api/list")
async def list_knowledge(limit: int = 50, offset: int = 0):
    try:
        results = collection.query(
            expr='id != ""',
            output_fields=["id", "content", "category", "tags", "timestamp"],
            limit=limit, offset=offset
        )
        db = get_db()
        c = db.cursor()
        items = []
        for r in results:
            c.execute("SELECT recall_count, last_recall_time, memory_level FROM memory_meta WHERE id = ?", (r["id"],))
            meta = c.fetchone()
            if meta:
                recall_count, last_recall_time, level = meta
                R = calc_retention(level, last_recall_time)
            else:
                recall_count, level, R = 0, "flash", 1.0
                ensure_meta(db, r["id"])
            items.append({
                "id": r.get("id"),
                "content": r.get("content"),
                "category": r.get("category"),
                "tags": r.get("tags", "").split(","),
                "time": datetime.fromtimestamp(r.get("timestamp", 0) / 1000).strftime("%Y-%m-%d %H:%M"),
                "memory_level": level,
                "recall_count": recall_count,
                "retention": round(R * 100, 2)
            })
        db.commit()
        db.close()
        return {"results": items, "total": collection.num_entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===================== 更新 =====================

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
        return {"message": "更新成功", "id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===================== 标记permanent =====================

@app.post("/api/set_level/{doc_id}")
async def set_memory_level(doc_id: str, level: str = "permanent"):
    if level not in STRENGTH:
        raise HTTPException(status_code=400, detail=f"无效层级: {level}")
    db = get_db()
    ensure_meta(db, doc_id)
    db.execute("UPDATE memory_meta SET memory_level = ? WHERE id = ?", (level, doc_id))
    db.commit()
    db.close()
    return {"message": f"已设为 {level}", "id": doc_id}

# ===================== 清理 =====================

@app.post("/api/cleanup")
async def cleanup_memories(req: CleanupRequest):
    db = get_db()
    c = db.cursor()
    c.execute("SELECT id, last_recall_time, memory_level FROM memory_meta WHERE memory_level != 'permanent'")
    rows = c.fetchall()
    deleted = 0
    for doc_id, lrt, level in rows:
        R = calc_retention(level, lrt)
        if R < req.threshold:
            collection.delete(expr=f'id == "{doc_id}"')
            c.execute("DELETE FROM memory_meta WHERE id = ?", (doc_id,))
            deleted += 1
    db.commit()
    db.close()
    if deleted > 0:
        collection.flush()
    return {"deleted": deleted, "message": f"清理了 {deleted} 条衰减记忆"}

# ===================== 日期/天气 =====================

@app.get("/api/today")
async def api_today():
    tz_cn = timezone(timedelta(hours=8))
    now = datetime.now(tz_cn).replace(tzinfo=None)
    a = cnlunar.Lunar(now, godType="8char")
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
    lunar_festivals = [a.get_legalHolidays(), a.get_otherHolidays()]
    festivals.extend([f for f in lunar_festivals if f])
    tomorrow = now + timedelta(days=1)
    a_tomorrow = cnlunar.Lunar(tomorrow, godType="8char")
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

async def fetch_weather(city: str) -> dict:
    if not QWEATHER_KEY:
        return {"error": "未配置天气API密钥"}
    try:
        async with httpx.AsyncClient() as client:
            geo_url = f"https://geoapi.qweather.com/v2/city/lookup?location={city}&key={QWEATHER_KEY}"
            geo_res = await client.get(geo_url)
            geo_data = geo_res.json()
            if geo_data.get("code") != "200" or not geo_data.get("location"):
                return {"error": f"找不到城市: {city}"}
            loc = geo_data["location"][0]
            city_id = loc["id"]
            city_name = loc["name"]
            now_url = f"https://devapi.qweather.com/v7/weather/now?location={city_id}&key={QWEATHER_KEY}"
            now_res = await client.get(now_url)
            now_data = now_res.json()
            if now_data.get("code") != "200":
                return {"error": "获取天气失败"}
            n = now_data["now"]
            forecast_url = f"https://devapi.qweather.com/v7/weather/3d?location={city_id}&key={QWEATHER_KEY}"
            forecast_res = await client.get(forecast_url)
            forecast_data = forecast_res.json()
            forecast = []
            if forecast_data.get("code") == "200":
                for d in forecast_data["daily"]:
                    forecast.append({"date": d["fxDate"], "textDay": d["textDay"], "textNight": d["textNight"], "tempMin": d["tempMin"], "tempMax": d["tempMax"]})
            return {"city": city_name, "temp": n["temp"], "feelsLike": n["feelsLike"], "text": n["text"], "humidity": n["humidity"], "windDir": n["windDir"], "windScale": n["windScale"], "forecast": forecast}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/weather")
async def api_weather(city: str = "北京"):
    return await fetch_weather(city)

# ===================== MCP =====================

mcp_server = FastMCP(
    "RecallDoggy",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False
    )
)

@mcp_server.tool()
async def mcp_search(query: str, top_k: int = 5) -> str:
    """搜索向量知识库，返回最相似的知识条目。permanent记忆始终返回。"""
    db = get_db()
    c = db.cursor()
    c.execute("SELECT id FROM memory_meta WHERE memory_level = 'permanent'")
    perm_ids = set(row[0] for row in c.fetchall())
    perm_items = []
    if perm_ids:
        id_list = '["' + '","'.join(perm_ids) + '"]'
        perm_results = collection.query(
            expr=f"id in {id_list}",
            output_fields=["content", "category", "tags"]
        )
        for r in perm_results:
            perm_items.append({
                "content": r["content"],
                "category": r["category"],
                "tags": r.get("tags", ""),
                "similarity": 100.0,
                "memory_level": "permanent"
            })
    query_vec = encoder.encode(query).tolist()
    results = collection.search(
        data=[query_vec], anns_field="embedding",
        param={"metric_type": "COSINE"}, limit=top_k * 2,
        output_fields=["content", "category", "tags", "timestamp"]
    )
    items = []
    for hits in results:
        for hit in hits:
            if hit.id in perm_ids:
                continue
            c.execute("SELECT recall_count, last_recall_time, memory_level FROM memory_meta WHERE id = ?", (hit.id,))
            meta = c.fetchone()
            if meta:
                recall_count, lrt, level = meta
                R = calc_retention(level, lrt)
            else:
                recall_count, level, R = 0, "flash", 1.0
                ensure_meta(db, hit.id)
            final_score = hit.score * 0.7 + R * 0.3
            items.append({
                "content": hit.entity.get("content"),
                "category": hit.entity.get("category"),
                "tags": hit.entity.get("tags", ""),
                "similarity": round(hit.score * 100, 2),
                "memory_level": auto_upgrade(recall_count, level),
                "retention": round(R * 100, 2),
                "_fs": final_score
            })
    items.sort(key=lambda x: x["_fs"], reverse=True)
    items = items[:top_k]
    now_ts = datetime.now().timestamp()
    for item in items:
        item.pop("_fs", None)
    all_items = perm_items + items
    db.commit()
    db.close()
    return json.dumps(all_items, ensure_ascii=False)

@mcp_server.tool()
async def mcp_write(content: str, category: str = "通用", tags: str = "") -> str:
    """向知识库写入一条新知识"""
    embedding = encoder.encode(content).tolist()
    doc_id = hashlib.md5(content.encode()).hexdigest()
    result = collection.query(expr=f'id == "{doc_id}"', output_fields=["id"], limit=1)
    if result:
        return "知识已存在"
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    tz_cn = timezone(timedelta(hours=8))
    time_tag = datetime.now(tz_cn).strftime("%Y-%m-%d %H:%M")
    tag_list.append(time_tag)
    data = [
        [doc_id], [embedding], [content], [category],
        [",".join(tag_list)], [int(datetime.now().timestamp() * 1000)]
    ]
    collection.insert(data)
    collection.flush()
    now = datetime.now().timestamp()
    db = get_db()
    db.execute("INSERT OR REPLACE INTO memory_meta (id, recall_count, last_recall_time, memory_level, created_at) VALUES (?, 0, ?, 'flash', ?)", (doc_id, now, now))
    db.commit()
    db.close()
    return f"写入成功，ID: {doc_id}"

@mcp_server.tool()
async def mcp_delete(doc_id: str) -> str:
    """从知识库删除指定ID的知识"""
    collection.delete(expr=f'id == "{doc_id}"')
    db = get_db()
    db.execute("DELETE FROM memory_meta WHERE id = ?", (doc_id,))
    db.commit()
    db.close()
    return f"已删除 {doc_id}"

@mcp_server.tool()
async def mcp_today() -> str:
    """获取今天的完整日期信息，包括公历、农历、节气、节日、纪念日。AI应在需要感知当前日期时主动调用此工具。"""
    tz_cn = timezone(timedelta(hours=8))
    now = datetime.now(tz_cn).replace(tzinfo=None)
    a = cnlunar.Lunar(now, godType="8char")
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
    lunar_festivals = [a.get_legalHolidays(), a.get_otherHolidays()]
    festivals.extend([f for f in lunar_festivals if f])
    tomorrow = now + timedelta(days=1)
    a_tomorrow = cnlunar.Lunar(tomorrow, godType="8char")
    if a_tomorrow.lunarMonthCn == "正月" and a_tomorrow.lunarDayCn == "初一":
        if "除夕" not in festivals:
            festivals.append("除夕")
    custom = []
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
        res = client.query(
            collection_name=COLLECTION_NAME,
            filter='category == "纪念日"',
            output_fields=["content", "tags"],
            limit=100
        )
        for item in res:
            tags = item.get("tags", [])
            if isinstance(tags, str):
                tags = tags.split(",")
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
    db = get_db()
    c = db.cursor()
    counts = {}
    for level in ["flash", "short", "long", "permanent"]:
        c.execute("SELECT COUNT(*) FROM memory_meta WHERE memory_level = ?", (level,))
        counts[level] = c.fetchone()[0]
    db.close()
    return json.dumps({"total": collection.num_entities, "collection": COLLECTION_NAME, "levels": counts}, ensure_ascii=False)

@mcp_server.tool()
async def mcp_weather(city: str = "天津") -> str:
    """查询指定城市的实时天气"""
    data = await fetch_weather(city)
    if "error" in data:
        return data["error"]
    lines = [f"🌡️ {data['city']}: {data['temp']}°C {data['text']}"]
    if data.get("forecast"):
        for d in data["forecast"]:
            lines.append(f"  {d['date']}: {d['textDay']} {d['tempMin']}~{d['tempMax']}°C")
    return "\n".join(lines)

app.mount("/mcp", mcp_server.sse_app())

if __name__ == "__main__":
    import sys
    if "--stdio" in sys.argv:
        mcp_server.run(transport="stdio")
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
