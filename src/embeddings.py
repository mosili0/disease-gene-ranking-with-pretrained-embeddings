# -*- coding: utf-8 -*-
"""Frozen models, special-token-free pooling, full ESM sequence coverage, validated caches."""
import hashlib
import json
import os
import re
from functools import lru_cache
import numpy as np
import torch
import config
from .utils import ensure_dir, write_json

EMB_DIR = os.path.join(config.DATA_DIR, "emb", "v2")
SCHEMA = "v3-full-coverage-nonoverlapping-1022-aa-and-254-wordpieces"


def _l2norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def _model_dir(name):
    return os.path.join(config.MODELS_DIR, config.MODEL_IDS[name].split("/")[-1])


@lru_cache(None)
def model_fingerprint(name):
    folder = _model_dir(name)
    files = [f for f in ["config.json", "vocab.txt", "tokenizer_config.json",
                         "model.safetensors", "pytorch_model.bin"] if os.path.isfile(os.path.join(folder, f))]
    out = {}
    for f in files:
        h = hashlib.sha256()
        with open(os.path.join(folder, f), "rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                h.update(chunk)
        out[f] = h.hexdigest()
    if not any(f in out for f in ["model.safetensors", "pytorch_model.bin"]):
        raise FileNotFoundError("missing model weights: " + folder)
    return out


def signature(name, inputs):
    obj = dict(schema=SCHEMA, model=config.MODEL_IDS[name], files=model_fingerprint(name), inputs=inputs)
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _read(cache, sig):
    if os.path.exists(cache):
        with np.load(cache) as z:
            if str(z["signature"]) == sig:
                return z["X"].copy(), z["order"].tolist()
    return None


def _save(cache, sig, X, order):
    tmp = cache + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, X=X, order=np.array(order), signature=np.array(sig))
    os.replace(tmp, cache)


def chunks(seq, size=1022):
    return [seq[i:i + size] for i in range(0, len(seq), size)]


class ESMEmbedder:
    def __init__(self, model_dir, device="cpu"):
        from transformers import EsmModel, EsmTokenizer
        self.tokenizer = EsmTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = EsmModel.from_pretrained(model_dir, local_files_only=True,
                                              add_pooling_layer=False).eval().to(device)
        self.model.requires_grad_(False)
        self.device = device

    @torch.inference_mode()
    def embed(self, seqs, batch_size=4):
        pieces, owners = [], []
        for i, seq in enumerate(seqs):
            for part in chunks(seq):
                pieces.append(part)
                owners.append(i)
        sums = np.zeros((len(seqs), self.model.config.hidden_size), dtype=np.float64)
        counts = np.zeros(len(seqs))
        # Length sorting reduces padding without changing the residue-weighted result.
        ix = sorted(range(len(pieces)), key=lambda i: len(pieces[i]))
        for start in range(0, len(ix), batch_size):
            ids = ix[start:start + batch_size]
            batch = [pieces[i] for i in ids]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=False,
                                 return_special_tokens_mask=True)
            special = enc.pop("special_tokens_mask").to(self.device)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            h = self.model(**enc).last_hidden_state
            mask = (enc["attention_mask"] * (1 - special)).unsqueeze(-1).float()
            weighted = (h * mask).sum(1).cpu().numpy()
            lengths = mask.sum(1).squeeze(-1).cpu().numpy()
            for j, piece_id in enumerate(ids):
                sums[owners[piece_id]] += weighted[j]
                counts[owners[piece_id]] += lengths[j]
        if not np.array_equal(counts, np.array([len(s) for s in seqs])):
            raise RuntimeError("ESM tokenizer did not cover each residue exactly once")
        return _l2norm((sums / counts[:, None]).astype("float32"))


class BioBERTEmbedder:
    def __init__(self, model_dir, device="cpu"):
        from transformers import BertModel, BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = BertModel.from_pretrained(model_dir, local_files_only=True).eval().to(device)
        self.model.requires_grad_(False)
        self.device = device

    @torch.inference_mode()
    def embed(self, texts, batch_size=8, max_length=256):
        # Match TF-IDF's full-text coverage. Chunking does not restore cross-window attention.
        pieces, owners, token_counts = [], [], []
        for i, text in enumerate(texts):
            ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            if not ids:
                ids = [self.tokenizer.unk_token_id]
            token_counts.append(len(ids))
            for start in range(0, len(ids), max_length - 2):
                pieces.append(ids[start:start + max_length - 2])
                owners.append(i)
        sums = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float64)
        counts = np.zeros(len(texts))
        order = sorted(range(len(pieces)), key=lambda i: len(pieces[i]))
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            width = max(len(pieces[i]) for i in ids) + 2
            tokens, attention, content = [], [], []
            for i in ids:
                n = len(pieces[i])
                pad = width - n - 2
                tokens.append([self.tokenizer.cls_token_id] + pieces[i] +
                              [self.tokenizer.sep_token_id] + [self.tokenizer.pad_token_id] * pad)
                attention.append([1] * (n+2) + [0] * pad)
                content.append([0] + [1] * n + [0] * (pad+1))
            enc = {"input_ids": torch.tensor(tokens, device=self.device),
                   "attention_mask": torch.tensor(attention, device=self.device)}
            h = self.model(**enc).last_hidden_state
            mask = torch.tensor(content, device=self.device).unsqueeze(-1).float()
            weighted = (h * mask).sum(1).cpu().numpy()
            for j, piece_id in enumerate(ids):
                sums[owners[piece_id]] += weighted[j]
                counts[owners[piece_id]] += len(pieces[piece_id])
        if not np.array_equal(counts, token_counts):
            raise RuntimeError("BioBERT token coverage mismatch")
        return _l2norm((sums / counts[:, None]).astype("float32"))


def gene_texts(gene_info, genes):
    texts = []
    for g in genes:
        d = gene_info.get(g, {})
        t = (d.get("summary") or d.get("name") or g).strip()
        # Remove database evidence tags/references for both BioBERT and TF-IDF.
        t = re.sub(r"\{[^}]*\}|PubMed:\s*\d+", " ", t, flags=re.I)
        texts.append(" ".join(t.split()))
    return texts


def load_all_embeddings(gene_info, seqs, models=None, device="cpu", force=False):
    ensure_dir(EMB_DIR)
    models = models or config.EMBED_MODELS
    gene_embs, meta = {}, dict(schema=SCHEMA, models={}, sequence_coverage=[])
    for name in models:
        order = [g for g in config.ALL_GENES if g in seqs] if name == "esm" else list(config.ALL_GENES)
        inputs = [seqs[g] for g in order] if name == "esm" else gene_texts(gene_info, order)
        sig = signature(name, list(zip(order, inputs)))
        cache = os.path.join(EMB_DIR, "genes_" + name + ".npz")
        saved = None if force else _read(cache, sig)
        if saved is None:
            print("[emb] computing corrected %s embeddings" % name, flush=True)
            embedder = ESMEmbedder(_model_dir(name), device) if name == "esm" else BioBERTEmbedder(_model_dir(name), device)
            X = embedder.embed(inputs)
            _save(cache, sig, X, order)
            if name == "biobert":
                lens = [len(embedder.tokenizer.encode(t, add_special_tokens=True)) for t in inputs]
                meta["text_truncation"] = dict(window_tokens=256, n_truncated=0,
                                                n_exceeding_single_window=sum(l > 256 for l in lens),
                                                token_lengths=dict(zip(order, lens)))
            del embedder
        else:
            X, order = saved
        gene_embs[name] = dict(zip(order, X))
        meta["models"][name] = dict(model_id=config.MODEL_IDS[name], n_genes=len(order), dim=X.shape[1],
                                     signature=sig, file_sha256=model_fingerprint(name))
    queries = config.all_queries()
    query_embs = {}
    if "biobert" in models:
        keys = list(queries)
        sig = signature("biobert", queries)
        cache = os.path.join(EMB_DIR, "queries_biobert.npz")
        saved = None if force else _read(cache, sig)
        if saved is None:
            embedder = BioBERTEmbedder(_model_dir("biobert"), device)
            X = embedder.embed([queries[k] for k in keys])
            _save(cache, sig, X, keys)
        else:
            X, keys = saved
        query_embs = dict(zip(keys, X))
        meta["query_signature"] = sig
    meta["sequence_coverage"] = [dict(gene=g, length=len(s), windows=len(chunks(s)),
                                       old_prefix_residues=min(len(s), 1022), new_covered_residues=len(s))
                                  for g, s in seqs.items()]
    # Retain deterministic token-length audit when loading existing validated embeddings.
    previous = os.path.join(EMB_DIR, "summary.json")
    if "text_truncation" not in meta and os.path.isfile(previous):
        old = json.load(open(previous, encoding="utf-8"))
        if old.get("models", {}).get("biobert", {}).get("signature") == meta["models"].get("biobert", {}).get("signature"):
            if "text_truncation" in old:
                meta["text_truncation"] = old["text_truncation"]
    write_json(previous, meta)
    return gene_embs, query_embs
