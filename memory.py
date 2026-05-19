"""分层记忆系统 - 核心常量与纯函数"""
import math
from datetime import datetime, timedelta, timezone

TZ_CN = timezone(timedelta(hours=8))

HALF_LIFE = {"flash": 24, "short": 168, "long": 720, "permanent": None}
UPGRADE_THRESHOLDS = {1: "short", 4: "long", 10: "permanent"}
LEVEL_ORDER = ["flash", "short", "long", "permanent"]


def now_ms():
    return int(datetime.now(TZ_CN).timestamp() * 1000)


def calc_retention(memory_level, last_recall_ts, recall_count=0):
    if memory_level == "permanent":
        return 1.0
    S = HALF_LIFE.get(memory_level, 24)
    t_hours = max(0, (now_ms() - last_recall_ts) / 3600000)
    return min(1.0, math.exp(-t_hours / S) * (max(1, recall_count) ** 0.3))


def check_upgrade(memory_level, recall_count):
    if memory_level == "permanent":
        return "permanent"
    for threshold in sorted(UPGRADE_THRESHOLDS.keys(), reverse=True):
        if recall_count >= threshold:
            target = UPGRADE_THRESHOLDS[threshold]
            if LEVEL_ORDER.index(target) > LEVEL_ORDER.index(memory_level):
                return target
    return memory_level


def format_item(r, similarity=None):
    ts = r.get("timestamp", 0)
    level = r.get("memory_level", "flash")
    rc = r.get("recall_count", 0)
    lr = r.get("last_recall", ts)
    ret = calc_retention(level, lr, rc)
    item = {
        "id": r.get("id"),
        "content": r.get("content"),
        "category": r.get("category"),
        "tags": r.get("tags", "").split(",") if isinstance(r.get("tags"), str) else r.get("tags", []),
        "time": datetime.fromtimestamp(ts / 1000, tz=TZ_CN).strftime("%Y-%m-%d %H:%M") if ts else "",
        "memory_level": level,
        "recall_count": rc,
        "retention": round(ret * 100, 2),
        "user": r.get("user", "default"),
    }
    if similarity is not None:
        item["similarity"] = similarity
    return item
