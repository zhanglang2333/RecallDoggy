---
name: recalldoggy
description: >
  当用户需要存储、搜索、管理知识条目时使用此skill。
  包括写入知识、语义搜索、列表、编辑、删除、查看统计、
  查询今日日期/农历/节气/纪念日。通过MCP工具操作。
---

# RecallDoggy 向量知识库

基于 FastAPI + Zilliz Cloud + SentenceTransformer 的知识库服务。

## MCP 连接方式

- SSE: "fix: remove server IP from SKILL.md"
- stdio: `python app.py --stdio`

## 工具列表

| 工具 | 功能 | 必填参数 | 可选参数 |
|------|------|----------|----------|
| mcp_write | 写入知识 | content(string) | category(string), tags(string[]) |
| mcp_search | 语义搜索 | query(string) | top_k(int, 默认5, 范围1-20) |
| mcp_delete | 删除条目 | id(int) | - |
| mcp_stats | 知识库统计 | - | - |
| mcp_today | 今日信息 | - | - |

## 使用场景

- "记一下xxx" / "记住xxx" → `mcp_write`
- "有没有关于xxx" / 搜索 → `mcp_search`
- "今天什么日子" → `mcp_today`
- "删掉那条" → `mcp_delete`

## 纪念日格式

category 设为 `纪念日`，content 格式：`名称|类型(公历/农历)|MM-DD`

## 注意事项

- 服务器UTC时区，代码内转北京时间(UTC+8)
- 嵌入模型：paraphrase-multilingual-MiniLM-L12-v2（支持中文）
