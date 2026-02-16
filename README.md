# 🐕 RecallDoggy

基于向量数据库的 AI 知识库系统，支持语义搜索、MCP 接入。

## ✨ 功能

- 📝 写入知识（支持分类、标签）
- 🔍 语义搜索（基于向量相似度，支持调整 top_k）
- 📋 知识列表（浏览、删除）
- 🤖 MCP SSE 接入（可连接 Claude 等 AI 客户端）

## 🛠️ 技术栈

- FastAPI
- Zilliz Cloud（Milvus 托管版）
- SentenceTransformer（paraphrase-multilingual-MiniLM-L12-v2）

## 🚀 Docker 一键部署

### 1.‌ 克隆仓库

```bash
git clone https://github.com/zhanglang2333/RecallDoggy.git
cd RecallDoggy

### 2.注册 Zilliz Cloud （免费的就行）

前往 [Zilliz Cloud](https://cloud.zilliz.com/) 注册账号，创建一个 Serverless 集群，获取连接地址和 Token。

### 3.配置环境变量

cp .env.example .env
编辑 `.env`，填入你的 Zilliz 连接信息
ZILLIZ_URI=https://你的地址.cloud.zilliz.com
ZILLIZ_TOKEN=你的token

### 4.启动
docker compose up -d --build

首次启动会下载模型，大约需要 3~5 分钟。
启动成功后访问：http://localhost:8000

### 5.停止

docker compose down
```

