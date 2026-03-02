---
name: recalldoggy
description: >
  当用户需要存储、搜索、管理知识条目时使用此skill。
  包括写入知识、语义搜索、列表、编辑、删除、查看统计、
  查询今日日期/农历/节气/纪念日、天气查询。
  支持分层记忆系统（flash/short/long/permanent），
  基于艾宾浩斯遗忘曲线自动衰减与巩固。通过MCP工具操作。
---

# RecallDoggy 向量知识库

基于 FastAPI + Zilliz Cloud(Milvus) + SentenceTransformer + SQLite 的分层记忆知识库服务。

## 架构

- **向量搜索**: Zilliz Cloud (Milvus) — 存储内容与嵌入向量
- **元数据管理**: SQLite — 存储记忆层级、召回次数、衰减状态
- **嵌入模型**: paraphrase-multilingual-MiniLM-L12-v2（本地，支持中文）
- **前端**: Jinja2 模板，主题色 #A0D8EF

## 分层记忆系统

### 记忆层级

| 层级 | 半衰期 | 触发条件 | 说明 |
|------|--------|---------|------|
| flash | 24小时 | 新写入 recall_count=0 | 没人搜就快速衰减 |
| short | 7天 | recall_count >= 1 | 被用过 保留一段时间 |
| long | 30天 | recall_count >= 4 | 反复巩固的知识 |
| permanent | 不衰减 | recall_count >= 10 或手动标记 | 核心信息永久保留 |

### 衰减公式（艾宾浩斯）

    R = e^(-t / S)

- t = 距上次recall的小时数
- S = 强度系数（flash=24, short=168, long=720, permanent=inf）
- R = 保留率 (0~1)

### 搜索加权

    final_score = similarity * 0.7 + retention * 0.3

### 自动升级

每次被搜索命中：recall_count += 1，刷新 last_recall_time，自动升级层级。

### permanent 特殊逻辑

- permanent 记忆不管搜什么都返回，不占 top_k 名额（置顶记忆）
- 纪念日写入自动设为 permanent
- 可通过前端 pin 按钮手动标记

## MCP 连接方式

- SSE: http://YOUR_SERVER_IP:8000/mcp/sse
- stdio: python app.py --stdio

## MCP 工具列表

| 工具 | 功能 | 必填参数 | 可选参数 |
|------|------|----------|----------|
| mcp_write | 写入知识 | content(string) | category(string), tags(string,逗号分隔) |
| mcp_search | 语义搜索（含permanent置顶） | query(string) | top_k(int, 默认5) |
| mcp_delete | 删除条目 | doc_id(string) | - |
| mcp_stats | 知识库统计（含各层级数量） | - | - |
| mcp_today | 今日日期/农历/节气/节日/纪念日 | - | - |
| mcp_weather | 查询城市天气 | - | city(string, 默认天津) |

## REST API 端点

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | / | 前端页面 |
| POST | /api/write | 写入知识（支持 memory_level 参数） |
| POST | /api/search | 语义搜索（返回 permanent + results） |
| GET | /api/stats | 统计信息（含 levels 分布） |
| DELETE | /api/delete/{id} | 删除知识 |
| GET | /api/list | 知识列表（含层级/召回次数/保留率） |
| PUT | /api/update/{id} | 更新知识内容 |
| GET | /api/today | 今日日期信息 |
| GET | /api/weather | 天气查询 |
| POST | /api/set_level/{id}?level=xxx | 手动设置记忆层级 |
| POST | /api/cleanup | 按阈值清理衰减记忆 |

## 使用场景

- 记一下xxx / 记住xxx -> mcp_write
- 有没有关于xxx / 搜索 -> mcp_search
- 今天什么日子 -> mcp_today
- 天气怎么样 -> mcp_weather
- 删掉那条 -> mcp_delete
- 知识库多少条了 -> mcp_stats

## 纪念日

- 写入时 category 设为「纪念日」，自动标记为 permanent
- tags 包含日期（如 02-14）和类型（solar/lunar）
- mcp_today 会自动匹配当日纪念日并返回

## 写入行为

- 自动追加时间标签到 tags（格式：YYYY-MM-DD HH:MM）
- 内容去重：相同内容不会重复写入（MD5校验）
- 默认层级 flash，可指定 memory_level

## 注意事项

- 服务器UTC时区，代码内转北京时间(UTC+8)
- 搜索时取 top_k*2 条候选，加权排序后截取 top_k
- permanent 记忆始终返回，不受 top_k 限制
- SQLite 文件: /root/metadata.db（运行时自动创建）
- 清理功能不会删除 permanent 记忆
