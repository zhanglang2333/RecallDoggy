"""HttpMemoryStore - 通过HTTP连接远程Milvus Lite API"""
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional
import httpx

from memory import (
    TZ_CN, LEVEL_ORDER,
    now_ms, calc_retention, check_upgrade, format_item
)
from store import MemoryStore, COLLECTION_NAME, EMBEDDING_DIM, ALL_FIELDS

logger = logging.getLogger("recalldoggy")


class HttpMemoryStore(MemoryStore):

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.client = httpx.AsyncClient(timeout=30)

    async def _post(self, path: str, json: dict) -> dict:
        r = await self.client.post(f"{self.base_url}{path}", json=json, headers=self.headers)
        r.raise_for_status()
        return r.json()

    async def _get(self, path: str) -> dict:
        r = await self.client.get(f"{self.base_url}{path}", headers=self.headers)
        r.raise_for_status()
        return r.json()

    async def _query(self, filter_expr: str, output_fields: list,
                     limit: int = 16384, offset: int = 0) -> list:
        res = await self._post("/query", {
            "collection_name": COLLECTION_NAME,
            "filter": filter_expr,
            "output_fields": output_fields,
            "limit": limit, "offset": offset,
        })
        return res.get("results", [])

    async def _insert(self, record: dict):
        await self._post("/insert", {
            "collection_name": COLLECTION_NAME, "data": [record],
        })

    async def _delete_expr(self, expr: str):
        await self._post("/delete", {
            "collection_name": COLLECTION_NAME, "filter": expr,
        })

    def _user_expr(self, user: str, extra: str = "") -> str:
        base = f'user == "{user}"'
        return f"{base} and {extra}" if extra else base

    async def connect(self) -> None:
        res = await self._get("/health")
        if res.get("status") != "ok":
            raise ConnectionError("Milvus API unreachable")
        logger.info("已连接远程 Milvus API")

        res = await self._get(f"/collection/has/{COLLECTION_NAME}")
        if not res.get("exists"):
            await self._create_collection()
        else:
            c = await self._get(f"/count/{COLLECTION_NAME}")
            logger.info(f"知识库就绪，当前: {c.get('count', '?')} 条")

    async def _create_collection(self):
        fields = [
            {"name": "id", "dtype": "VARCHAR", "is_primary": True, "max_length": 64},
            {"name": "embedding", "dtype": "FLOAT_VECTOR", "dim": EMBEDDING_DIM},
            {"name": "content", "dtype": "VARCHAR", "max_length": 10000},
            {"name": "category", "dtype": "VARCHAR", "max_length": 100},
            {"name": "tags", "dtype": "VARCHAR", "max_length": 500},
            {"name": "timestamp", "dtype": "INT64"},
            {"name": "memory_level", "dtype": "VARCHAR", "max_length": 20},
            {"name": "recall_count", "dtype": "INT64"},
            {"name": "last_recall", "dtype": "INT64"},
            {"name": "user", "dtype": "VARCHAR", "max_length": 64},
        ]
        await self._post("/collection/create_schema", {
            "collection_name": COLLECTION_NAME,
            "fields": fields,
            "index_field": "embedding",
            "metric_type": "COSINE",
        })
        logger.info("远程collection已创建")

    async def write(self, content, embedding, category, tags, memory_level, user="default"):
        doc_id = hashlib.md5(content.encode()).hexdigest()
        existing = await self._query(f'id == "{doc_id}"', ["id"], limit=1)
        if existing:
            return {"status": "exists", "message": "知识已存在"}
        level = memory_level if memory_level in LEVEL_ORDER else "flash"
        ts = now_ms()
        await self._insert({
            "id": doc_id, "embedding": embedding, "content": content,
            "category": category,
            "tags": ",".join(tags) if isinstance(tags, list) else tags,
            "timestamp": ts, "memory_level": level,
            "recall_count": 0, "last_recall": ts, "user": user,
        })
        logger.info(f"写入[{level}]: {content[:50]} | {category} | user={user}")
        return {"status": "success", "message": "写入成功", "id": doc_id}

    async def search(self, query_vec, top_k, user="default", update_recall=True):
        perm_expr = self._user_expr(user, 'memory_level == "permanent"')
        perm_raw = await self._query(perm_expr, ALL_FIELDS, limit=100)
        permanent = [format_item(r) for r in perm_raw]
        perm_ids = {r["id"] for r in perm_raw}

        res = await self._post("/search", {
            "collection_name": COLLECTION_NAME,
            "data": [query_vec],
            "limit": top_k + len(perm_raw),
            "output_fields": ["content", "category", "tags", "timestamp",
                              "memory_level", "recall_count", "last_recall", "user"],
            "filter": self._user_expr(user),
        })
        hits_raw = res.get("results", [[]])[0]

        scored = []
        for hit in hits_raw:
            if hit["id"] in perm_ids:
                continue
            e = hit.get("entity", {})
            level = e.get("memory_level", "flash")
            lr = e.get("last_recall", e.get("timestamp", 0))
            rc = e.get("recall_count", 0)
            ret = calc_retention(level, lr, rc)
            sim = hit.get("distance", 0)
            scored.append({
                "id": hit["id"], "entity": e,
                "final_score": sim * 0.7 + ret * 0.3,
                "retention": ret, "similarity": sim,
            })

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        scored = scored[:top_k]

        if update_recall:
            for it in scored:
                try: await self.do_recall(it["id"])
                except: pass
            for r in perm_raw:
                try: await self.do_recall(r["id"])
                except: pass

        output = []
        for it in scored:
            e = it["entity"]
            output.append({
                "id": it["id"], "content": e.get("content"),
                "category": e.get("category"),
                "tags": e.get("tags", "").split(","),
                "similarity": round(it["similarity"] * 100, 2),
                "time": datetime.fromtimestamp(
                    e.get("timestamp", 0) / 1000, tz=TZ_CN
                ).strftime("%Y-%m-%d %H:%M"),
                "memory_level": e.get("memory_level", "flash"),
                "recall_count": e.get("recall_count", 0) + 1,
                "retention": round(it["retention"] * 100, 2),
            })

        logger.info(f"搜索: top_k={top_k} | 结果:{len(output)} | permanent:{len(permanent)} | user={user}")
        return {"results": output, "permanent": permanent}

    async def delete(self, doc_id):
        await self._delete_expr(f'id == "{doc_id}"')
        logger.warning(f"删除: {doc_id}")
        return True

    async def update(self, doc_id, content, embedding, category, tags):
        r = await self.get_by_id(doc_id)
        if not r:
            return None
        await self._delete_expr(f'id == "{doc_id}"')
        await self._insert({
            "id": doc_id, "embedding": embedding, "content": content,
            "category": category,
            "tags": ",".join(tags) if isinstance(tags, list) else tags,
            "timestamp": r.get("timestamp", now_ms()),
            "memory_level": r.get("memory_level", "flash"),
            "recall_count": r.get("recall_count", 0),
            "last_recall": r.get("last_recall", now_ms()),
            "user": r.get("user", "default"),
        })
        logger.info(f"更新: {doc_id}")
        return {"message": "更新成功", "id": doc_id}

    async def get_by_id(self, doc_id):
        results = await self._query(f'id == "{doc_id}"', ALL_FIELDS, limit=1)
        return results[0] if results else None

    async def list_all(self, limit, offset, user="default"):
        results = await self._query(self._user_expr(user), ALL_FIELDS, limit=limit, offset=offset)
        c = await self._get(f"/count/{COLLECTION_NAME}")
        return {"results": [format_item(r) for r in results], "total": c.get("count", 0)}

    async def set_level(self, doc_id, level):
        if level not in LEVEL_ORDER:
            return None
        r = await self._query(f'id == "{doc_id}"', ALL_FIELDS + ["embedding"], limit=1)
        if not r:
            return None
        r = r[0]
        await self._delete_expr(f'id == "{doc_id}"')
        await self._insert({
            "id": r["id"], "embedding": r["embedding"],
            "content": r["content"], "category": r["category"],
            "tags": r["tags"], "timestamp": r["timestamp"],
            "memory_level": level, "recall_count": r.get("recall_count", 0),
            "last_recall": now_ms(), "user": r.get("user", "default"),
        })
        logger.info(f"层级变更: {doc_id} -> {level}")
        return {"message": f"已设为 {level}", "id": doc_id}

    async def do_recall(self, doc_id):
        r = await self._query(f'id == "{doc_id}"', ALL_FIELDS + ["embedding"], limit=1)
        if not r:
            return
        r = r[0]
        old_level = r.get("memory_level", "flash")
        new_count = r.get("recall_count", 0) + 1
        new_level = check_upgrade(old_level, new_count)
        await self._delete_expr(f'id == "{doc_id}"')
        await self._insert({
            "id": r["id"], "embedding": r["embedding"],
            "content": r["content"], "category": r["category"],
            "tags": r["tags"], "timestamp": r["timestamp"],
            "memory_level": new_level, "recall_count": new_count,
            "last_recall": now_ms(), "user": r.get("user", "default"),
        })
        if new_level != old_level:
            logger.info(f"记忆升级: {doc_id} {old_level} -> {new_level} (recall={new_count})")

    async def cleanup(self, threshold, user="default"):
        expr = self._user_expr(user, 'memory_level != "permanent"')
        all_data = await self._query(expr, ALL_FIELDS, limit=16384)
        to_delete = [
            r["id"] for r in all_data
            if calc_retention(r.get("memory_level", "flash"),
                             r.get("last_recall", r.get("timestamp", 0)),
                             r.get("recall_count", 0)) < threshold
        ]
        for doc_id in to_delete:
            await self._delete_expr(f'id == "{doc_id}"')
        logger.info(f"清理: 删除{len(to_delete)}条 | 阈值{threshold} | user={user}")
        return len(to_delete)

    async def stats(self, user="default"):
        all_data = await self._query(self._user_expr(user),
                                     ["memory_level", "content", "category", "timestamp"], limit=16384)
        levels = {"flash": 0, "short": 0, "long": 0, "permanent": 0}
        now = datetime.now(TZ_CN)
        trend = {(now - timedelta(days=i)).strftime("%m-%d"): 0 for i in range(6, -1, -1)}
        for r in all_data:
            levels[r.get("memory_level", "flash")] = levels.get(r.get("memory_level", "flash"), 0) + 1
            ts = r.get("timestamp", 0)
            if ts:
                key = datetime.fromtimestamp(ts / 1000, tz=TZ_CN).strftime("%m-%d")
                if key in trend:
                    trend[key] += 1
        sorted_data = sorted(all_data, key=lambda x: x.get("timestamp", 0), reverse=True)[:10]
        recent = [{"category": r.get("category", ""), "content": r.get("content", "")[:80],
                   "time": datetime.fromtimestamp(r.get("timestamp", 0) / 1000, tz=TZ_CN).strftime("%m-%d %H:%M")
                   if r.get("timestamp") else ""} for r in sorted_data]
        return {"total": len(all_data), "collection": COLLECTION_NAME,
                "levels": levels, "trend": trend, "recent": recent}

    async def dashboard(self, user="default"):
        all_data = await self._query(self._user_expr(user), ALL_FIELDS, limit=16384)
        levels = {"flash": 0, "short": 0, "long": 0, "permanent": 0}
        cat_count = {}
        now = datetime.now(TZ_CN)
        trend = {(now - timedelta(days=i)).strftime("%m-%d"): 0 for i in range(6, -1, -1)}
        for item in all_data:
            lv = item.get("memory_level", "flash")
            levels[lv] = levels.get(lv, 0) + 1
            cat = item.get("category", "未分类")
            cat_count[cat] = cat_count.get(cat, 0) + 1
            ts = item.get("timestamp", 0)
            if ts:
                key = datetime.fromtimestamp(ts / 1000, tz=TZ_CN).strftime("%m-%d")
                if key in trend:
                    trend[key] += 1
        sorted_data = sorted(all_data, key=lambda x: x.get("timestamp", 0), reverse=True)
        recent = [{"id": r.get("id"), "content": r.get("content", "")[:80],
                   "category": r.get("category"), "memory_level": r.get("memory_level", "flash"),
                   "time": datetime.fromtimestamp(r.get("timestamp", 0) / 1000, tz=TZ_CN).strftime("%m-%d %H:%M")}
                  for r in sorted_data[:10]]
        return {"total": len(all_data), "levels": levels,
                "categories": cat_count, "trend": trend, "recent": recent}

    async def export_all(self, user="default"):
        all_data = await self._query(self._user_expr(user), ALL_FIELDS, limit=16384)
        return [format_item(r) for r in all_data]

    async def count(self):
        res = await self._get(f"/count/{COLLECTION_NAME}")
        return res.get("count", 0)

    async def query_by_category(self, category, fields, limit, user="default"):
        expr = self._user_expr(user, f'category == "{category}"')
        return await self._query(expr, fields, limit=limit)
