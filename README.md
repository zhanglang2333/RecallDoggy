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
```

### 4. 启动服务

**SSE 模式（远程部署）：**

```bash
python app.py
```

服务启动在 `http://0.0.0.0:8000`，MCP 端点为 `/sse`

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
      "url": "http://你的服务器IP:8000/sse"
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
docker run -d -p 8000:8000 --env-file .env recalldoggy
```

## 🛠️ MCP 工具列表

| 工具 | 说明 |
|---|---|
| `mcp_write` | 写入记忆 |
| `mcp_search` | 语义搜索记忆 |
| `mcp_list` | 列出所有记忆 |
| `mcp_edit` | 编辑记忆 |
| `mcp_delete` | 删除记忆 |
| `mcp_add_anniversary` | 添加纪念日 |
| `mcp_search_anniversary` | 查询纪念日 |
| `mcp_delete_anniversary` | 删除纪念日 |
| `mcp_today` | 获取今日信息（农历/节气/节日/纪念日） |

## 📋 环境要求

| 依赖 | 版本 |
|---|---|
| Python | 3.10+ |
| 操作系统 | Windows / macOS / Linux |

## 📄 License

MIT