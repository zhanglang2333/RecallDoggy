---
name: recalldoggy
description: >
  当用户需要存储、搜索、管理知识条目时使用此skill。
  包括写入知识、语义搜索、列表、编辑、删除、查看统计、
  查询今日日期/农历/节气/纪念日、天气查询。
  支持分层记忆系统（flash/short/long/permanent），
  基于艾宾浩斯遗忘曲线自动衰减与巩固。
  支持多用户记忆隔离（user字段）。通过MCP工具操作。
---

# RecallDoggy 向量知识库

基于 FastAPI + Zilliz Cloud(Milvus) + SentenceTransformer 的分层记忆知识库服务。

## 架构

- **代码拆分**: memory.py（纯计算）+ store.py（MemoryStore 抽象基类 + ZillizMemoryStore）+ app.py（路由 + 中间件 + MCP）
- **向量搜索**: Zilliz Cloud (Milvus) — 存储内容、嵌入向量、记忆层级、召回次数、衰减状态、用户标识
- **嵌入模型**: paraphrase-multilingual-MiniLM-L12-v2（本地，支持中文）
- **前端**: Jinja2 模板，主题色 #A0D8EF
- **认证**: bcrypt + session + AuthMiddleware，登录限流（滑动窗口）

## 多用户记忆隔离

Zilliz schema 包含 `user` 字段（VARCHAR 64），所有写入/搜索/统计/导出自动按 user 过滤。

| user 值 | 用途 |
|---------|------|
| default | 旧数据迁移默认值 |
| claude | Claude 的记忆 |
| 4o | GPT-4o 的记忆 |

## 分层记忆系统

### 记忆层级

| 层级 | 半衰期 | 触发条件 | 说明 |
|------|--------|---------|------|
| flash | 24小时 | 新写入 recall_count=0 | 没人搜就快速衰减 |
| short | 7天 | recall_count >= 1 | 被用过 保留一段时间 |
| long | 30天 | recall_count >= 4 | 反复巩固的知识 |
| permanent | 不衰减 | recall_count >= 10 或手动标记 | 核心信息永久保留 |

### 衰减公式（艾宾浩斯）

    R = min(1.0, e^(-t / S) × recall_count^0.3)

- t = 距上次recall的小时数
- S = 强度系数（flash=24, short=168, long=720, permanent=inf）
- R = 保留率 (0~1)
- recall_count=0时按1计算（不影响新记忆）
- 被召回越多衰减越慢，上限cap在1.0

### 搜索加权

    final_score = similarity * 0.7 + retention * 0.3

### 自动升级

每次被搜索命中：recall_count += 1，刷新 last_recall，自动升级层级。

### permanent 特殊逻辑

- permanent 记忆不管搜什么都返回，不占 top_k 名额（置顶记忆）
- 纪念日写入自动设为 permanent
- 可通过前端 pin 按钮手动标记

## MCP 连接方式

| 传输模式 | 端点 | 认证 |
|---------|------|------|
| SSE | /mcp/sse | Bearer Token |
| Streamable HTTP | /mcp-http/mcp | Bearer Token |
| stdio | python app.py --stdio | 无需认证 |

## MCP 工具列表

| 工具 | 功能 | 必填参数 | 可选参数 |
|------|------|----------|----------|
| mcp_write | 写入知识 | content(string) | category(string, 仅限：通用/技术/生活/学习/纪念日/人物/项目), tags(string,逗号分隔纯文本), memory_level(string, flash/short/long/permanent), user(string) |
| mcp_search | 语义搜索（含permanent置顶） | query(string, 提取核心关键词，不要丢完整对话原文) | top_k(int, 默认5), user(string) |
| mcp_delete | 删除条目 | doc_id(string, 从搜索结果id字段获取，不要编造) | - |
| mcp_stats | 知识库统计（含各层级数量） | - | user(string) |
| mcp_today | 今日日期/农历/节气/节日/纪念日 | - | - |
| mcp_weather | 查询城市天气 | city(string, 必填，不要假设默认城市) | - |

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
| GET | /health | 健康检查 |
| GET | /api/export | 导出全部记忆 |
| GET | /dashboard | Dashboard 页面 |
| GET | /logs | 日志页面 |
| GET | /api/logs | 日志 API |
| GET | /api/dashboard | Dashboard 数据 API |
| POST | /api/update-system | 一键更新（执行 update.sh） |

## 使用场景

- 记一下xxx / 记住xxx -> mcp_write
- 有没有关于xxx / 搜索 -> mcp_search（提取关键词搜索，不要丢整句话）
- 今天什么日子 -> mcp_today
- xx城市天气怎么样 -> mcp_weather（必须指定城市，不知道就问用户）
- 删掉那条 -> mcp_delete（先搜索拿到id再删）
- 知识库多少条了 -> mcp_stats

## 纪念日

- 写入时 category 设为「纪念日」，自动标记为 permanent
- tags 包含日期（如 02-14）和类型（solar/lunar），用逗号分隔纯文本
- mcp_today 会自动匹配当日纪念日并返回

## 写入行为

- 内容去重：相同内容不会重复写入（MD5校验）
- 默认层级 flash，可指定 memory_level
- 默认 user 为 "default"，可指定 user 隔离记忆
- category 只接受固定值：通用/技术/生活/学习/纪念日/人物/项目

## 注意事项

- 服务器UTC时区，代码内转北京时间(UTC+8)
- 搜索时 permanent 先单独查出置顶，剩余走向量搜索加权排序
- permanent 记忆始终返回，不受 top_k 限制
- 清理功能不会删除 permanent 记忆
- 旧数据（无 user 字段）启动时自动迁移为 user="default"
- tags 只接受逗号分隔纯文本，不要传 JSON 数组
