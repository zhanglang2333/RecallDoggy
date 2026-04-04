"""数据库抽象层 - MemoryStore基类 + ZillizMemoryStore实现"""
import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

from pymilvus import (
    connections, Collection, FieldSchema,
    CollectionSchema, DataType, utility
)

from memory import (
    TZ_CN, LEVEL_ORDER,
    now_ms, calc_retention, check_upgrade, format_item
)

logger = logging.getLogger("recalldoggy")

COLLECTION_NAME = "ai_knowledge"
EMBEDDING_DIM = 384
ALL_FIELDS = [
    "id", "content", "category", "tags", "timestamp",
    "memory_level", "recall_count", "last_recall", "user"
]


class MemoryStore(ABC):

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def write(self, content: str, embedding: list, category: str,
                    tags: list, memory_level: str, user: str = "default") -> dict: ...

    @abstractmethod
    async def search(self, query_vec: list, top_k: int,
                     user: str = "default", update_recall: bool = True) -> dict: ...

    @abstractmethod
    async def delete(self, doc_id: str) -> bool: ...

    @abstractmethod
    async def update(self, doc_id: str, content: str, embedding: list,
                     category: str, tags: list) -> dict: ...

    @abstractmethod
    async def get_by_id(self, doc_id: str) -> Optional[dict]: ...

    @abstractmethod
    async def list_all(self, limit: int, offset: int,
                       user: str = "default") -> dict: ...

    @abstractmethod
    async def set_level(self, doc_id: str, level: str) -> dict: ...

    @abstractmethod
    async def cleanup(self, threshold: float, user: str = "default") -> int: ...

    @abstractmethod
    async def do_recall(self, doc_id: str) -> None: ...

    @abstractmethod
    async def stats(self, user: str = "default") -> dict: ...

    @abstractmethod
    async def dashboard(self, user: str = "default") -> dict: ...

    @abstractmethod
    async def export_all(self, user: str = "default") -> list: ...

    @abstractmethod
    async def count(self) -> int: ...

    @abstractmethod
    async def query_by_category(self, category: str, fields: list,
                                limit: int, user: str = "default") -> list: ...


class ZillizMemoryStore(MemoryStore):

    def __init__(self, uri: str, token: str):
        self.uri = uri
        self.token = token
        self.collection: Optional[Collection] = None

    async def connect(self) -> None:
        connections.connect(alias="default", uri=self.uri, token=self.token)
        logger.info("已连接 Zilliz")

        if utility.has_collection(COLLECTION_NAME):
            old = Collection(COLLECTION_NAME)
            field_names = [f.name for f in old.schema.fields]

            if "memory_level" not in field_names:
                logger.warning("旧schema(无memory_level)，直接重建...")
                utility.drop_collection(COLLECTION_NAME)

            elif "user" not in field_names:
                logger.warning("检测到旧schema(无user字段)，开始迁移...")
                await self._migrate_add_user(old)
                return

            else:
                self.collection = old
                self.collection.load()
                logger.info(f"知识库就绪，当前: {self.collection.num_entities} 条")
                return

        self._create_collection()

    async def _migrate_add_user(self, old_col: Collection):
        old_col.load()
        all_data = old_col.query(
            expr='id != ""',
            output_fields=[
                "id", "embedding", "content", "category", "tags",
                "timestamp", "memory_level", "recall_count", "last_recall"
            ],
            limit=16384
        )
        count = len(all_data)
        logger.info(f"导出 {count} 条记录用于迁移")

        utility.drop_collection(COLLECTION_NAME)
        self._create_collection()

        if all_data:
            batch_size = 100
            for i in range(0, count, batch_size):
                batch = all_data[i:i + batch_size]
                data = [
                    [r["id"] for r in batch],
                    [r["embedding"] for r in batch],
                    [r["content"] for r in batch],
                    [r["category"] for r in batch],
                    [r.get("tags", "") for r in batch],
                    [r.get("timestamp", now_ms()) for r in batch],
                    [r.get("memory_level", "flash") for r in batch],
                    [r.get("recall_count", 0) for r in batch],
                    [r.get("last_recall", now_ms()) for r in batch],
                    ["default" for _ in batch],
                ]
                self.collection.insert(data)
            self.collection.flush()
        logger.info(f"迁移完成: {count} 条 -> user=default")

    def _create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="memory_level", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="recall_count", dtype=DataType.INT64),
            FieldSchema(name="last_recall", dtype=DataType.INT64),
            FieldSchema(name="user", dtype=DataType.VARCHAR, max_length=64),
        ]
        schema = CollectionSchema(fields=fields)
        self.collection = Collection(name=COLLECTION_NAME, schema=schema)
        self.collection.create_index(
            field_name="embedding",
            index_params={"metric_type": "COSINE", "index_type": "AUTOINDEX", "params": {}}
        )
        self.collection.load()
        logger.info("新collection已创建（含user字段）")

    def _user_expr(self, user: str, extra: str = "") -> str:
        base = f'user == "{user}"'
        return f"{base} and {extra}" if extra else base

    def _insert_one(self, rec: dict):
        data = [
            [rec["id"]], [rec["embedding"]], [rec["content"]],
            [rec["category"]], [rec["tags"]], [rec["timestamp"]],
            [rec["memory_level"]], [rec["recall_count"]],
            [rec["last_recall"]], [rec["user"]],
        ]
        self.collection.insert(data)

    def _get_with_embedding(self, doc_id: str) -> Optional[dict]:
        results = self.collection.query(
            expr=f'id == "{doc_id}"',
            output_fields=ALL_FIELDS + ["embedding"],
            limit=1
        )
        return results[0] if results else None

    async def write(self, content, embedding, category, tags, memory_level, user="default"):
        doc_id = hashlib.md5(content.encode()).hexdigest()
        if self.collection.query(expr=f'id == "{doc_id}"', output_fields=["id"], limit=1):
            return {"status": "exists", "message": "知识已存在"}

        level = memory_level if memory_level in LEVEL_ORDER else "flash"
        ts = now_ms()
        self._insert_one({
            "id": doc_id, "embedding": embedding, "content": content,
            "category": category,
            "tags": ",".join(tags) if isinstance(tags, list) else tags,
            "timestamp": ts, "memory_level": level,
            "recall_count": 0, "last_recall": ts, "user": user,
        })
        self.collection.flush()
        logger.info(f"写入[{level}]: {content[:50]} | {category} | user={user}")
        return {"status": "success", "message": "写入成功", "id": doc_id}

    async def search(self, query_vec, top_k, user="default", update_recall=True):
        perm_expr = self._user_expr(user, 'memory_level == "permanent"')
        perm_raw = self.collection.query(expr=perm_expr, output_fields=ALL_FIELDS, limit=100)
        permanent = [format_item(r) for r in perm_raw]
        perm_ids = {r["id"] for r in perm_raw}

        hits_raw = self.collection.search(
            data=[query_vec], anns_field="embedding",
            param={"metric_type": "COSINE"},
            limit=top_k + len(perm_raw),
            output_fields=[
                "content", "category", "tags", "timestamp",
                "memory_level", "recall_count", "last_recall", "user"
            ],
            expr=self._user_expr(user),
        )

        scored = []
        for hits in hits_raw:
            for hit in hits:
                if hit.id in perm_ids:
                    continue
                level = hit.entity.get("memory_level", "flash")
                lr = hit.entity.get("last_recall", hit.entity.get("timestamp", 0))
                rc = hit.entity.get("recall_count", 0)
                ret = calc_retention(level, lr, rc)
                sim = hit.score
                scored.append({
                    "hit": hit,
                    "final_score": sim * 0.7 + ret * 0.3,
                    "retention": ret,
                    "similarity": sim,
                })

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        scored = scored[:top_k]

        if update_recall:
            for it in scored:
                try:
                    await self.do_recall(it["hit"].id)
                except Exception:
                    pass
            for r in perm_raw:
                try:
                    await self.do_recall(r["id"])
                except Exception:
                    pass

        output = []
        for it in scored:
            h = it["hit"]
            output.append({
                "id": h.id,
                "content": h.entity.get("content"),
                "category": h.entity.get("category"),
                "tags": h.entity.get("tags", "").split(","),
                "similarity": round(it["similarity"] * 100, 2),
                "time": datetime.fromtimestamp(
                    h.entity.get("timestamp", 0) / 1000, tz=TZ_CN
                ).strftime("%Y-%m-%d %H:%M"),
                "memory_level": h.entity.get("memory_level", "flash"),
                "recall_count": h.entity.get("recall_count", 0) + 1,
                "retention": round(it["retention"] * 100, 2),
            })

        logger.info(
            f"搜索: top_k={top_k} | 结果:{len(output)} | "
            f"permanent:{len(permanent)} | user={user}"
        )
        return {"results": output, "permanent": permanent}

    async def delete(self, doc_id):
        self.collection.delete(expr=f'id == "{doc_id}"')
        logger.warning(f"删除: {doc_id}")
        return True

    async def update(self, doc_id, content, embedding, category, tags):
        r = await self.get_by_id(doc_id)
        if not r:
            return None
        self.collection.delete(expr=f'id == "{doc_id}"')
        self._insert_one({
            "id": doc_id, "embedding": embedding, "content": content,
            "category": category,
            "tags": ",".join(tags) if isinstance(tags, list) else tags,
            "timestamp": r.get("timestamp", now_ms()),
            "memory_level": r.get("memory_level", "flash"),
            "recall_count": r.get("recall_count", 0),
            "last_recall": r.get("last_recall", now_ms()),
            "user": r.get("user", "default"),
        })
        self.collection.flush()
        logger.info(f"更新: {doc_id}")
        return {"message": "更新成功", "id": doc_id}

    async def get_by_id(self, doc_id):
        results = self.collection.query(
            expr=f'id == "{doc_id}"', output_fields=ALL_FIELDS, limit=1
        )
        return results[0] if results else None

    async def list_all(self, limit, offset, user="default"):
        results = self.collection.query(
            expr=self._user_expr(user),
            output_fields=ALL_FIELDS, limit=limit, offset=offset
        )
        return {
            "results": [format_item(r) for r in results],
            "total": self.collection.num_entities
        }

    async def set_level(self, doc_id, level):
        if level not in LEVEL_ORDER:
            return None
        r = self._get_with_embedding(doc_id)
        if not r:
            return None
        self.collection.delete(expr=f'id == "{doc_id}"')
        self._insert_one({
            "id": r["id"], "embedding": r["embedding"],
            "content": r["content"], "category": r["category"],
            "tags": r["tags"], "timestamp": r["timestamp"],
            "memory_level": level,
            "recall_count": r.get("recall_count", 0),
            "last_recall": now_ms(),
            "user": r.get("user", "default"),
        })
        self.collection.flush()
        logger.info(f"层级变更: {doc_id} -> {level}")
        return {"message": f"已设为 {level}", "id": doc_id}

    async def do_recall(self, doc_id):
        r = self._get_with_embedding(doc_id)
        if not r:
            return
        old_level = r.get("memory_level", "flash")
        new_count = r.get("recall_count", 0) + 1
        new_level = check_upgrade(old_level, new_count)

        self.collection.delete(expr=f'id == "{doc_id}"')
        self._insert_one({
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
        all_data = self.collection.query(expr=expr, output_fields=ALL_FIELDS, limit=16384)
        to_delete = [
            r["id"] for r in all_data
            if calc_retention(
                r.get("memory_level", "flash"),
                r.get("last_recall", r.get("timestamp", 0)),
                r.get("recall_count", 0)
            ) < threshold
        ]
        for doc_id in to_delete:
            self.collection.delete(expr=f'id == "{doc_id}"')
        logger.info(f"清理: 删除{len(to_delete)}条 | 阈值{threshold} | user={user}")
        return len(to_delete)

    async def stats(self, user="default"):
        expr = self._user_expr(user)
        all_data = self.collection.query(
            expr=expr,
            output_fields=["memory_level", "content", "category", "timestamp"],
            limit=16384
        )
        levels = {"flash": 0, "short": 0, "long": 0, "permanent": 0}
        now = datetime.now(TZ_CN)
        trend = {(now - timedelta(days=i)).strftime("%m-%d"): 0 for i in range(6, -1, -1)}

        for r in all_data:
            lv = r.get("memory_level", "flash")
            levels[lv] = levels.get(lv, 0) + 1
            ts = r.get("timestamp", 0)
            if ts:
                key = datetime.fromtimestamp(ts / 1000, tz=TZ_CN).strftime("%m-%d")
                if key in trend:
                    trend[key] += 1

        sorted_data = sorted(all_data, key=lambda x: x.get("timestamp", 0), reverse=True)[:10]
        recent = [{
            "category": r.get("category", ""),
            "content": r.get("content", "")[:80],
            "time": datetime.fromtimestamp(
                r.get("timestamp", 0) / 1000, tz=TZ_CN
            ).strftime("%m-%d %H:%M") if r.get("timestamp") else ""
        } for r in sorted_data]

        return {
            "total": len(all_data), "collection": COLLECTION_NAME,
            "levels": levels, "trend": trend, "recent": recent,
        }

    async def dashboard(self, user="default"):
        expr = self._user_expr(user)
        all_data = self.collection.query(expr=expr, output_fields=ALL_FIELDS, limit=16384)

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
        recent = [{
            "id": r.get("id"),
            "content": r.get("content", "")[:80],
            "category": r.get("category"),
            "memory_level": r.get("memory_level", "flash"),
            "time": datetime.fromtimestamp(
                r.get("timestamp", 0) / 1000, tz=TZ_CN
            ).strftime("%m-%d %H:%M"),
        } for r in sorted_data[:10]]

        return {
            "total": len(all_data), "levels": levels,
            "categories": cat_count, "trend": trend, "recent": recent,
        }

    async def export_all(self, user="default"):
        expr = self._user_expr(user)
        all_data = self.collection.query(expr=expr, output_fields=ALL_FIELDS, limit=16384)
        return [format_item(r) for r in all_data]

    async def count(self):
        return self.collection.num_entities if self.collection else 0

    async def query_by_category(self, category, fields, limit, user="default"):
        expr = self._user_expr(user, f'category == "{category}"')
        return self.collection.query(expr=expr, output_fields=fields, limit=limit)
