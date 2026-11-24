# -*- coding: utf-8 -*-
"""通用工具：安全的文件读写（自动重试，规避沙箱对只读临时目录的写入限制）。"""
import json
import os
import time
import math

RETRY = 6
SLEEP = 1.5


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _retry(fn, *args, **kwargs):
    for attempt in range(RETRY):
        try:
            return fn(*args, **kwargs)
        except PermissionError:
            if attempt == RETRY - 1:
                raise
            time.sleep(SLEEP)


def write_json(path, obj):
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"

    def _w():
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_json_safe(obj), f, ensure_ascii=False, indent=1, allow_nan=False)
        os.replace(tmp, path)

    _retry(_w)
    return path


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path, text):
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"

    def _w():
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)

    _retry(_w)
    return path
