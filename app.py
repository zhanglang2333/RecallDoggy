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
from dotenv import load_dotenv
load_dotenv()
from typing import List
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

encoder = None
collection = None

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
    print("启动服务...")
    connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    print("已连接 Zilliz")
    encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
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
    print(f"知识库就绪，当前知识数：{collection.num_entities}")

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
        return {"results": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def stats_api():
    return {"total": collection.num_entities, "collection": COLLECTION_NAME}

#@mcp_server.tool()
#async def mcp_weather(city: str = "北京") -> str:
#    """查询指定城市的实时天气和未来3天预报"""
#    data = await fetch_weather(city)
#    if "error" in data:
#        return data["error"]
#    lines = [f"🌡️ {data['city']}：{data['text']}，{data['temp']}°C（体感{data['feelsLike']}°C），湿度{data['humidity']}%，{data['windDir']}{data['windScale']}级"]
#    if data.get("forecast"):
#        for d in data["forecast"]:
#            lines.append(f"  {d['date']}：{d['textDay']}→{d['textNight']}，{d['tempMin']}~{d['tempMax']}°C")
#    return "\n".join(lines)

@app.delete("/api/delete/{doc_id}")
async def delete_knowledge(doc_id: str):
    try:
        collection.delete(expr=f'id == "{doc_id}"')
        return {"status": "success", "message": "已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

# ===================== MCP =====================

@app.get("/api/today")
async def api_today():
    """前端用：获取今日日期信息"""
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
    
    lunar_festivals = [a.get_legalHolidays(), a.get_otherHolidays()]
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
async def fetch_weather(city:str) ->dict:
    if not QWEATHER_KEY:
        return{"error":"未配置天气API密钥"}
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
    
    lunar_festivals = [a.get_legalHolidays(), a.get_otherHolidays()]
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

if __name__ == "__main__":
    import sys
    if "--stdio" in sys.argv:
        mcp_server.run(transport="stdio")
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
