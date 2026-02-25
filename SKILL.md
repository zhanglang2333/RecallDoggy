---
name: recalldoggy
description: >
  个人向量知识库助手。支持知识写入/语义搜索/管理/时间感知/天气查询。
  通过 MCP 工具操作，轻量部署，低 token 消耗。
---

# RecallDoggy

个人向量知识库，基于 FastAPI + Zilliz Cloud(Milvus) + SentenceTransformer。

## 连接

| 方式 | 地址 |
|------|------|
| SSE | `http://<HOST>:8000/mcp/sse` |
| stdio | `python app.py --stdio` |

## 工具

### mcp_write
写入一条知识到向量库。
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| content | string | ✅ | 知识内容 |
| category | string | | 分类（如：笔记、纪念日） |
| tags | string[] | | 标签列表 |

> 纪念日写入格式：category 设为 `纪念日`，content 为 `名称|类型(公历/农历)|MM-DD`

### mcp_search
语义搜索知识库，返回最相似的条目。
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| query | string | ✅ | 搜索内容 |
| top_k | int | | 返回数量，默认 5，范围 1-20 |

### mcp_delete
按 ID 删除一条知识。
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| doc_id | int | ✅ | 条目 ID |

### mcp_today
获取今日完整时间信息，无需参数。
- 返回：公历日期、农历、节气、节日、知识库中的纪念日

### mcp_stats
查看知识库统计信息，无需参数。

### mcp_weather
查询城市实时天气及未来预报。
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| city | string | | 城市名，默认 `天津` |

## 意图映射

| 用户说 | 调用 |
|--------|------|
| "记一下…" / "记住…" | mcp_write |
| "有没有关于…" / "搜一下…" | mcp_search |
| "今天几号" / "什么日子" | mcp_today |
| "天气怎么样" | mcp_weather |
| "删掉那条" | mcp_delete |
| "知识库多少条了" | mcp_stats |

## 技术备注

- 时区：服务器 UTC，代码内转北京时间（UTC+8）
- 嵌入模型：`paraphrase-multilingual-MiniLM-L12-v2`（本地推理，支持中文）
- 向量维度：384
- 部署：systemd 服务，支持 Docker
