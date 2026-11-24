# -*- coding: utf-8 -*-
"""Portable, revision-pinned model downloader; dependencies use standard pip."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config


def download(url, dest):
    part = str(dest) + ".part"
    for attempt in range(4):
        try:
            start = os.path.getsize(part) if os.path.isfile(part) else 0
            headers = {"User-Agent": "bioinfo-coursework/2.0"}
            if start:
                headers["Range"] = "bytes=%d-" % start
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                # A server returning 200 ignored Range: restart instead of appending a duplicate file.
                mode = "ab" if start and response.status == 206 else "wb"
                with open(part, mode) as stream:
                    while True:
                        block = response.read(1 << 20)
                        if not block:
                            break
                        stream.write(block)
            os.replace(part, dest)
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and os.path.isfile(part):
                os.remove(part)
            elif attempt == 3:
                raise
        except OSError:
            if attempt == 3:
                raise
        time.sleep(2)
    raise RuntimeError("download failed: " + url)


def install_models(name):
    repo = config.MODEL_IDS[name]
    out = ROOT / "models" / repo.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen("https://huggingface.co/api/models/" + repo, timeout=30) as r:
        info = json.load(r)
    revision = info["sha"]
    available = {x["rfilename"] for x in info["siblings"]}
    weight = "model.safetensors" if "model.safetensors" in available else "pytorch_model.bin"
    files = [f for f in ["config.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json", weight]
             if f in available]
    existed = [f for f in files if (out / f).exists()]
    for f in files:
        if (out / f).exists():
            print("[bootstrap] cached " + name + "/" + f, flush=True)
            continue
        print("[bootstrap] downloading " + name + "/" + f, flush=True)
        download("https://huggingface.co/%s/resolve/%s/%s" % (repo, revision, f), out / f)
    record = dict(repository=repo, download_revision=revision,
                  existing_files_not_redownloaded=existed,
                  note="Existing files have unknown original revision; fingerprints below identify exact local bytes.",
                  sha256={})
    for f in files:
        h = hashlib.sha256()
        with open(out / f, "rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                h.update(chunk)
        record["sha256"][f] = h.hexdigest()
    (out / "download_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["esm", "biobert", "all", "deps", "wheels"], default="all", nargs="?")
    stage = p.parse_args().stage
    if stage in ["deps", "wheels"]:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
    else:
        for name in config.MODEL_IDS if stage == "all" else [stage]:
            install_models(name)


if __name__ == "__main__":
    main()
