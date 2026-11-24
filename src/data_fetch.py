# -*- coding: utf-8 -*-
"""
真实数据获取与缓存（全部来自 UniProt，公开数据库，数据可追溯）：

  * 基因 -> UniProt Swiss-Prot 编号：按官方基因符号批量检索（reviewed=Swiss-Prot）；
  * 基因注释：蛋白全名、功能描述文本（cc_function，作为基因的文本描述）、
    GO 生物学过程注释（go_p，作为基因功能金标准）；
  * 蛋白序列：规范序列 FASTA（用于 ESM-2 蛋白语言模型）。

数据只获取一次并缓存到 data/ 目录，之后完全离线可复现。
"""
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

import config
from .utils import ensure_dir, write_json, write_text

UA = {"User-Agent": "bioinfo-coursework/1.0"}
SLEEP = 1.5
MAX_RETRIES = 3
UNIPROT = "https://rest.uniprot.org/uniprotkb"
OFFLINE = False


def http_get(url, timeout=30):
    """GET with certificate verification and bounded retries."""
    if OFFLINE:
        raise RuntimeError("offline mode: UniProt cache is missing")
    last = None
    for _ in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=dict(UA))
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionResetError) as e:
            last = e
            time.sleep(SLEEP)
    raise last


def _tsv_search(query, fields, size=500, timeout=60):
    """UniProt TSV 检索。返回按“请求字段顺序”解析的字典列表
    （TSV 表头是显示标签，如 Entry/Gene Names (primary)，与字段名不一致）。"""
    field_list = [f.strip() for f in fields.split(",") if f.strip()]
    url = ("%s/search?format=tsv&size=%d&fields=%s&query=%s"
           % (UNIPROT, size, urllib.parse.quote(",".join(field_list)),
              urllib.parse.quote(query, safe="():")))
    text = http_get(url, timeout=timeout).decode("utf-8")
    lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    rows = [ln.split("\t") for ln in lines[1:]]
    out = []
    for r in rows:
        d = {}
        for i, f in enumerate(field_list):
            d[f] = r[i] if i < len(r) else ""
        out.append(d)
    return out


def symbol2accession(symbols, timeout=90):
    """官方基因符号 -> Swiss-Prot 编号（批量，一次请求）。"""
    query = "(" + " OR ".join("gene:%s" % s for s in symbols) + \
        ") AND (organism_id:9606) AND (reviewed:true)"
    mapping = {}
    for d in _tsv_search(query, "accession,gene_primary", timeout=timeout):
        g = (d.get("gene_primary") or "").upper()
        g = {v: k for k, v in config.GENE_SYMBOL_ALIASES.items()}.get(g, g)
        if g:
            mapping.setdefault(g, d.get("accession", ""))
    return mapping


def fetch_gene_info(symbols, force=False):
    """返回 {symbol: {name, summary, uniprot, go_bp}}（UniProt 数据缓存到 data/）。"""
    out = os.path.join(config.DATA_DIR, "gene_info.json")
    if os.path.exists(out) and not force:
        return json.load(open(out, encoding="utf-8"))

    # 1) 符号 -> 编号
    acc_of = symbol2accession(symbols)
    missing_sym = [s for s in symbols if s not in acc_of]
    if missing_sym:
        print("[data] reviewed:true missed %d genes, retrying without filter: %s"
              % (len(missing_sym), missing_sym), flush=True)
        q2 = "(" + " OR ".join("gene:%s" % s for s in missing_sym) + \
            ") AND (organism_id:9606)"
        for d in _tsv_search(q2, "accession,gene_primary", timeout=90):
            g = (d.get("gene_primary") or "").upper()
            g = {v: k for k, v in config.GENE_SYMBOL_ALIASES.items()}.get(g, g)
            if g and g not in acc_of:
                acc_of[g] = d.get("accession", "")

    # 2) 注释批量获取
    accs = [acc_of[s] for s in symbols if acc_of.get(s)]
    annot = {}
    for d in _tsv_search(
            " OR ".join("accession:%s" % a for a in accs),
            "accession,gene_primary,protein_name,cc_function,go_p", timeout=90):
        annot[d.get("accession", "")] = d

    info = {}
    for sym in symbols:
        acc = acc_of.get(sym)
        d = annot.get(acc, {})
        go_bp = [g.strip() for g in (d.get("go_p") or "").split(";") if g.strip()]
        summary = (d.get("cc_function") or "").strip()
        info[sym] = {
            "name": (d.get("protein_name") or sym).split(" (")[0],
            "summary": summary,
            "uniprot": [acc] if acc else [],
            "go_bp": go_bp,
        }
    write_json(out, info)
    print("[data] uniprot annotations: %d/%d genes mapped, %d with GO-BP, %d with function text"
          % (len([s for s in symbols if info[s]["uniprot"]]), len(symbols),
             len([s for s in symbols if info[s]["go_bp"]]),
             len([s for s in symbols if info[s]["summary"]])), flush=True)
    return info


def fetch_sequences(gene_info, force=False):
    """按 UniProt 编号抓取规范蛋白序列，返回 {gene: sequence}。"""
    fasta_path = os.path.join(config.DATA_DIR, "sequences.fasta")
    seqs_path = os.path.join(config.DATA_DIR, "sequences.json")
    if os.path.exists(fasta_path) and os.path.exists(seqs_path) and not force:
        return json.load(open(seqs_path, encoding="utf-8"))

    acc2gene = {}
    for g, d in gene_info.items():
        if d["uniprot"]:
            acc2gene[d["uniprot"][0]] = g
    accs = list(acc2gene.keys())

    seq_by_acc = {}
    query = " OR ".join("accession:%s" % a for a in accs)
    try:
        fasta = http_get("%s/stream?format=fasta&query=%s"
                         % (UNIPROT, urllib.parse.quote(query)), timeout=60)
        seq_by_acc = parse_fasta(fasta.decode("utf-8"))
    except Exception as e:
        print("[data] uniprot fasta batch failed (%s), falling back per-gene" % e,
              flush=True)

    for a in [x for x in accs if x not in seq_by_acc]:
        try:
            fa = http_get("%s/%s.fasta" % (UNIPROT, a), timeout=30)
            seq_by_acc.update(parse_fasta(fa.decode("utf-8")))
        except Exception as e:
            print("[data] uniprot %s failed: %s" % (a, e), flush=True)
        time.sleep(0.2)

    seqs = {g: seq_by_acc[a] for a, g in acc2gene.items() if a in seq_by_acc}
    write_text(fasta_path, "\n".join(">%s\n%s" % (a, seq_by_acc[a])
                                     for a in sorted(seq_by_acc)))
    write_json(seqs_path, seqs)
    print("[data] uniprot sequences: %d/%d genes" % (len(seqs), len(acc2gene)),
          flush=True)
    return seqs


def parse_fasta(text):
    """极简 FASTA 解析，返回 {acc: sequence}。"""
    out, acc, buf = {}, None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if acc:
                out[acc] = "".join(buf)
            acc = line[1:].split("|")[1] if "|" in line else line[1:].split()[0]
            buf = []
        elif acc:
            buf.append(line.strip())
    if acc:
        out[acc] = "".join(buf)
    return out


def ensure_data(force=False):
    """确保 gene_info.json 与 sequences.json 存在（首次运行需联网）。"""
    ensure_dir(config.DATA_DIR)
    gene_info = fetch_gene_info(config.ALL_GENES, force=force)
    seqs = fetch_sequences(gene_info, force=force)
    return gene_info, seqs
