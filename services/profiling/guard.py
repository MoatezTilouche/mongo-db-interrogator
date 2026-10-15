"""A closed read API: raw database objects and cursors never escape the guard."""
from itertools import islice
import math
import time


class BudgetExceeded(TimeoutError):
    pass


def _check_expression(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"$out", "$merge", "$where", "$function", "$accumulator"}:
                raise PermissionError(f"Forbidden scan expression: {key}")
            _check_expression(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _check_expression(child)


def _consume(cursor, limit):
    try:
        return list(islice(cursor, limit))
    finally:
        close = getattr(cursor, "close", None)
        if close:
            close()


class ReadOnlyCollection:
    __slots__ = ("__collection", "__guard")

    def __init__(self, collection, guard):
        self.__collection, self.__guard = collection, guard

    def __getattr__(self, name):
        raise PermissionError(f"Scan operation is not allowed: {name}")

    def estimated_document_count(self):
        # Scalar result: intrinsically bounded to one integer.
        return self.__collection.estimated_document_count(maxTimeMS=self.__guard.time_limit())

    def indexes(self):
        return self.__guard.indexes(self.__collection.name)

    def find(self, predicate, *, limit, projection=None, sort=None, hint=None):
        limit = self.__guard.bound(limit)
        _check_expression(predicate)
        _check_expression(projection)
        kwargs = {"limit": limit, "max_time_ms": self.__guard.time_limit()}
        if sort:
            kwargs["sort"] = sort
        if hint:
            kwargs["hint"] = hint
        return _consume(self.__collection.find(predicate, projection, **kwargs), limit)

    def aggregate(self, pipeline, *, limit, hint=None):
        limit = self.__guard.bound(limit)
        allowed = {"$match", "$sample", "$group", "$project", "$limit", "$sort", "$count"}
        for stage in pipeline:
            if not isinstance(stage, dict) or len(stage) != 1 or not set(stage) <= allowed:
                raise PermissionError("Scan aggregation stage is not allowed")
            if "$sample" in stage:
                self.__guard.bound(stage["$sample"]["size"])
            if "$limit" in stage:
                self.__guard.bound(stage["$limit"])
        _check_expression(pipeline)
        kwargs = {"maxTimeMS": self.__guard.time_limit(), "allowDiskUse": False,
                  "batchSize": limit}
        if hint:
            kwargs["hint"] = hint
        return _consume(self.__collection.aggregate([*pipeline, {"$limit": limit}], **kwargs), limit)


class ReadOnlyDatabase:
    __slots__ = ("__database", "max_time_ms", "size_limit", "deadline", "clock", "metadata_truncated")

    def __init__(self, database, *, max_time_ms, size_limit, deadline, clock=time.monotonic):
        self.__database = database
        self.max_time_ms, self.size_limit = max_time_ms, size_limit
        self.deadline, self.clock = deadline, clock
        self.metadata_truncated = False

    def __getattr__(self, name):
        raise PermissionError(f"Scan operation is not allowed: {name}")

    def time_limit(self):
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise BudgetExceeded("Total scan budget exhausted")
        return min(self.max_time_ms, max(1, math.ceil(remaining * 1000)))

    def bound(self, size):
        if not isinstance(size, int) or not 0 < size <= self.size_limit:
            raise ValueError("A positive bounded scan size is required")
        return size

    def __getitem__(self, name):
        self.time_limit()
        return ReadOnlyCollection(self.__database[name], self)

    def collection_names(self):
        # listCollections cannot take a server-side limit. Bound cursor batches
        # and consumption; the scanner reports hitting the metadata cap.
        return [item["name"] for item in self.__metadata("listCollections", 1, nameOnly=True)]

    def indexes(self, collection):
        return self.__metadata("listIndexes", collection)

    def __metadata(self, operation, target, **options):
        # PyMongo's list_indexes has no maxTimeMS argument. Use a fixed command
        # instead, consume just the bounded initial batch, and close any cursor.
        if operation not in {"listCollections", "listIndexes"}:
            raise PermissionError("Scan metadata operation is not allowed")
        result = self.__database.command(operation, target, cursor={"batchSize": self.size_limit},
                                         maxTimeMS=self.time_limit(), **options)
        cursor = result["cursor"]
        if cursor.get("id"):
            self.metadata_truncated = True
            namespace = cursor["ns"].split(".", 1)[1]
            self.__database.command("killCursors", namespace, cursors=[cursor["id"]],
                                     maxTimeMS=self.time_limit())
        return cursor["firstBatch"][:self.size_limit]

    def connection_privileges(self):
        # The sole command exposed is read-only and has one response document.
        return self.__database.command("connectionStatus", showPrivileges=True,
                                       maxTimeMS=self.time_limit())
