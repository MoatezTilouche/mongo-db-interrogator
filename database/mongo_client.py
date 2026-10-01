from __future__ import annotations

from functools import lru_cache

from agent.config import MONGODB_DATABASE, MONGODB_URI


@lru_cache(maxsize=1)
def get_client():
    from pymongo import MongoClient
    return MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)


def get_database():
    if not MONGODB_DATABASE:
        raise ValueError("MONGODB_DATABASE is required")
    return get_client()[MONGODB_DATABASE]


def ping_database() -> bool:
    get_client().admin.command("ping")
    return True
