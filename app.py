from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse
from pymilvus import connections, Collection,FiedlSchema, CollectionSchema,DataType, utility 
from sentence_transformers import SentenceTransformer 
from pydantic import BaseModel 
import asyncio 
import json 
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
# 模板设置^~^
         #• •#
        #< ~ >
templates = Jinja2Templates(directory="templates")
# ==================== 
# 配置（修改这里！）====================
ZILLIZ_URI = "https://in03-0beca05df8c2566.serverless.aws-eu-central-1.cloud.zilliz.com" 
ZILLIZ_TOKEN = "b41a64341024ca7e783593b5af4bfcec49497e20d96517d397a2cc5f24842b62a0488a4f8edfec19b026ad3354a9e17bfbacfa87" 
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
@app.on_event("startup") 
async def startup():
    global encoder, collection
    
    print("🚀 启动服务...")
    
    connections.connect( 
        alias="default", 
        uri=ZILLIZ_URI, 
        token=ZILLIZ_TOKEN
    )
    print("✅ 已连接 Zilliz")
    
    print("📦加载模型（首次需要下载）...") 
    encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') 
    print("✅ 模型加载成功")
    
    if utility.has_collection(COLLECTION_NAME):
        collection = Collection(COLLECTION_NAME)
    else: fields = [ 
            FieldSchema(name="id", 
            dtype=DataType.VARCHAR, 
            is_primary=True, 
            max_length=64), 
            FieldSchema(name="embedding", 
            dtype=DataType.FLOAT_VECTOR, 
            dim=EMBEDDING_DIM), 
            FieldSchema(name="content", 
            dtype=DataType.VARCHAR, 
            max_length=10000), 
            FieldSchema(name="category", 
            dtype=DataType.VARCHAR, 
            max_length=100), 
            FieldSchema(name="tags", 
            dtype=DataType.VARCHAR, 
            max_length=500), 
            FieldSchema(name="timestamp", 
            dtype=DataType.INT64),
        ]
        
    schema = CollectionSchema(fields=fields) 
    collection = Collection(name=COLLECTION_NAME, schema=schema)
        
    index_params = { 
            "metric_type": 
            "COSINE", "index_type": 
            "AUTOINDEX", "params": 
            {}
        }
    collection.create_index(field_name="embedding", 
    index_params=index_params)
    
    collection.load() 
    print(f"知识库就绪，当前知识数：{collection.num_entities}")
# ==================== Web UI 
# ====================
@app.get("/", response_class=HTMLResponse) 
async def home(request: Request):
    """主页 - Web UI""" 
    return 
    templates.TemplateResponse("index.html", 
    {
        "request": request, 
        "total": 
        collection.num_entities if 
        collection else 0
    })
# ==================== API 
# ====================
@app.post("/api/write") 
async def write_knowledge(req: WriteRequest):
    try: embedding =  encoder.encode(req.content).tolist() 
        doc_id = hashlib.md5(req.content.encode()).hexdigest()
        
        result = collection.query( 
            expr=f'id == "{doc_id}"', 
            output_fields=["id"], 
            limit=1
        )
        
        if result: 
return {"status": "exists", "message": "知识已存在"}
        
        data = [ [doc_id], 
            [embedding], 
            [req.content], 
            [req.category], 
            [",".join(req.tags)], 
            [int(datetime.now().timestamp() 
            * 1000)]
        ]
        
        collection.insert(data) 
        collection.flush()
        
        return { "status": 
            "success", "message": 
            "✅ 写入成功", "id": 
            doc_id
        }
    except Exception as e: 
raise HTTPException(status_code=500, 
        detail=str(e))
@app.post("/api/search") async def 
search_knowledge(req: 
SearchRequest):
    try: query_vec = encoder.encode(req.query).tolist()
        
        results = collection.search(
            data=[query_vec], 
            anns_field="embedding", 
            param={"metric_type": 
            "COSINE"}, 
            limit=req.top_k, 
            output_fields=["content", 
            "category", "tags", 
            "timestamp"]
        )
        
        items = [] for hits in 
        results:
            for hit in hits: 
                items.append({
                    "id": hit.id, 
                    "content": 
                    hit.entity.get("content"), 
                    "category": 
                    hit.entity.get("category"), 
                    "tags": 
                    hit.entity.get("tags", 
                    "").split(","), 
                    "similarity": 
                    round(hit.score 
                    * 100, 2), 
                    "time": 
                    datetime.fromtimestamp(
                        hit.entity.get("timestamp", 
                        0) / 1000
                    ).strftime("%Y-%m-%d %H:%M")
                })
        
        return {"results": items} 
    except Exception as e:
        raise 
        HTTPException(status_code=500, 
        detail=str(e))
@app.get("/api/stats") async def 
stats():
    return { "total": 
        collection.num_entities, 
        "collection": 
        COLLECTION_NAME
    }
@app.delete("/api/delete/{doc_id}") 
async def delete_knowledge(doc_id: 
str):
    try: collection.delete(expr=f'id == "{doc_id}"') return 
        {"status": "success", 
        "message": "已删除"}
    except Exception as e: raise 
        HTTPException(status_code=500, 
        detail=str(e))
if __name__ == "__main__": import 
    uvicorn uvicorn.run(app, 
    host="0.0.0.0", port=8000)
