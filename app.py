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
from datetime import datetime
import hashlib

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
async def mcp_stats() -> str:
    """查看知识库统计信息"""
    return json.dumps({"total": collection.num_entities, "collection": COLLECTION_NAME})

app.mount("/mcp", mcp_server.sse_app())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
