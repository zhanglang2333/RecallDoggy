# 🐕 RecallDoggy

一个基于 Zilliz Cloud 向量数据库的 MCP 知识库服务，支持 SSE 和 stdio 双传输模式。

## ✨ 功能特性

- 📝 **记忆管理** - 写入 / 语义搜索 / 列表 / 编辑 / 删除
- 🎉 **纪念日管理** - 添加 / 查询 / 删除纪念日
- 📅 **时间感知** - mcp_today 工具，支持农历、节气、节日、纪念日查询
- 🌐 **双传输模式** - SSE（远程部署）+ stdio（本地直连）
- 🐳 **Docker 支持** - 一键容器化部署
- 🖥️ **前端页面** - 可视化管理知识库
- 🧠 **分层记忆** - flash/short/long/permanent 四级记忆 + 艾宾浩斯衰减
- 🌤️ **天气查询** - mcp_weather 工具

## 🧠 分层记忆系统

| 层级 | 半衰期 | 升级条件 | 说明 |
|------|--------|---------|------|
| 🔴 flash | 24小时 | 默认 | 新写入，没人搜就快速遗忘 |
| 🟡 short | 7天 | recall ≥ 1 | 被用过，保留一段时间 |
| 🟢 long | 30天 | recall ≥ 4 | 反复巩固的知识 |
| 💎 permanent | ∞ | recall ≥ 10 或手动📌 | 核心记忆，永不遗忘 |

- 衰减公式：`R = e^(-t/S)`（t=小时数，S=强度系数）
- 搜索加权：`final_score = similarity × 0.7 + retention × 0.3`
- permanent 记忆不管搜什么都会返回，不占 top_k 名额

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/zhanglang2333/RecallDoggy.git
cd RecallDoggy
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

创建 `.env` 文件：

```env
ZILLIZ_URI=你的Zilliz Cloud地址
ZILLIZ_TOKEN=你的Zilliz Cloud Token
HEFENG_API_KEY=你的和风天气API Key（可选）
```

> SQLite 数据库（metadata.db）运行时自动创建，无需配置。

### 4. 启动服务

**SSE 模式（远程部署）：**

```bash
python app.py
```

服务启动在 `http://0.0.0.0:8001`，MCP 端点为 `/sse`

**stdio 模式（本地直连）：**

```bash
python app.py --stdio
```

## 🔧 MCP 客户端配置

### SSE 模式

在 MCP 客户端中添加：

```json
{
  "mcpServers": {
    "RecallDoggy": {
      "url": "http://你的服务器IP:8001/sse"
    }
  }
}
```

### stdio 模式

**Mac / Linux：**

```json
{
  "mcpServers": {
    "RecallDoggy": {
      "command": "python3",
      "args": ["/path/to/RecallDoggy/app.py", "--stdio"],
      "env": {
        "ZILLIZ_URI": "你的uri",
        "ZILLIZ_TOKEN": "你的token"
      }
    }
  }
}
```

**Windows：**

```json
{
  "mcpServers": {
    "RecallDoggy": {
      "command": "python",
      "args": ["C:\\path\\to\\RecallDoggy\\app.py", "--stdio"],
      "env": {
        "ZILLIZ_URI": "你的uri",
        "ZILLIZ_TOKEN": "你的token"
      }
    }
  }
}
```

## 🐳 Docker 部署

```bash
docker build -t recalldoggy .
docker run -d -p 8001:8001 --env-file .env recalldoggy
```

## 健康检查

```
GET /health
```

无需登录，返回服务状态。

## 数据导出

```
GET /api/export
```

需登录，导出全部记忆为JSON。

## 更新

```bash
bash update.sh
```

# 🛠️ MCP 工具列表

| 工具 | 功能 | 必填参数 | 可选参数 |
|------|------|----------|----------|
| `mcp_write` | 写入记忆 | content | category, tags |
| `mcp_search` | 语义搜索（含 permanent 置顶） | query | top_k（默认5） |
| `mcp_delete` | 删除记忆 | doc_id | - |
| `mcp_stats` | 知识库统计（含各层级数量） | - | - |
| `mcp_today` | 今日信息（农历/节气/节日/纪念日） | - | - |
| `mcp_weather` | 天气查询 | - | city（默认天津） |

（因为我是天津人所以默认的天津(￣^￣)ゞ）

## 📋 环境要求

| 依赖 | 版本 |
|---|---|
| Python | 3.10+ |
| 操作系统 | Windows / macOS / Linux |

## 📄 License

MIT