"""Value-based learned-state hashing, stable across joblib array reconstruction.

Hash the complete estimator attribute tree without pickle memo/alias identity.
Object identity and fit-generation guards remain separate runtime checks.
Nothing is written and no parameter values are printed or returned.
"""

from dataclasses import is_dataclass, fields
from hashlib import sha256
from types import FunctionType

import numpy as np

from face_recognition.domain.models.rejection import RejectionError


def fitted_value_sha256(value):
    digest = sha256()
    active = set()
    def token(kind, content=b""):
        digest.update(kind + str(len(content)).encode("ascii") + b":" + content)
    def visit(item):
        if item is None: token(b"none")
        elif type(item) is bool: token(b"bool", str(item).encode("ascii"))
        elif type(item) is int: token(b"int", str(item).encode("ascii"))
        elif type(item) is float: token(b"float", item.hex().encode("ascii"))
        elif isinstance(item, str): token(b"str", item.encode("utf-8"))
        elif isinstance(item, np.ndarray):
            if item.dtype.hasobject: raise RejectionError("Object arrays are not supported in fitted V3 state")
            token(b"array", (item.dtype.str + repr(item.shape)).encode("ascii"))
            contiguous = np.ascontiguousarray(item)
            token(b"bytes-length", str(contiguous.nbytes).encode("ascii"))
            if contiguous.nbytes:  # probability=False SVC contains empty arrays.
                raw = memoryview(contiguous).cast("B")
                for start in range(0, raw.nbytes, 1024 * 1024): digest.update(raw[start:start + 1024 * 1024])
        elif isinstance(item, np.generic):
            token(b"scalar", item.dtype.str.encode("ascii") + item.tobytes())
        elif isinstance(item, FunctionType):
            token(b"function", (item.__module__ + "." + item.__qualname__).encode("utf-8"))
        else:
            if id(item) in active: raise RejectionError("Cyclic fitted state is not supported")
            active.add(id(item))
            try:
                if isinstance(item, (tuple, list)):
                    token(b"tuple" if isinstance(item, tuple) else b"list", str(len(item)).encode("ascii"))
                    for child in item: visit(child)
                elif isinstance(item, dict):
                    if any(not isinstance(k, str) for k in item): raise RejectionError("Invalid fitted attribute keys")
                    token(b"dict", str(len(item)).encode("ascii"))
                    for key in sorted(item): visit(key); visit(item[key])
                elif is_dataclass(item) or hasattr(item, "__dict__"):
                    token(b"object", (type(item).__module__ + "." + type(item).__qualname__).encode("utf-8"))
                    visit({f.name: getattr(item, f.name) for f in fields(item)} if is_dataclass(item) else vars(item))
                else: raise RejectionError("Unsupported fitted attribute type")
            finally: active.remove(id(item))
    visit(value)
    return digest.hexdigest()
