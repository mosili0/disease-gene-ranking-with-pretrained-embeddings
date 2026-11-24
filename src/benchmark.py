# -*- coding: utf-8 -*-
"""Complete candidate-universe annotations, provenance and fold-matched baselines."""
import json
import os
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import config
from .tasks import evaluate_entity
from .utils import ensure_dir, read_json, write_json

BENCH_DIR = os.path.join(config.DATA_DIR, "benchmark")
ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"
DISEASE_MONDO = dict(alzheimer="MONDO_0004975", parkinson="MONDO_0005180",
                     breast_cancer="MONDO_0007254", colorectal_cancer="MONDO_0005575",
                     lung_cancer="MONDO_0005233")
OT_SCORE_THRESHOLD = 0.2
OFFLINE = False


def today():
    return datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()


def http_json(url, data=None, timeout=40):
    if OFFLINE:
        raise RuntimeError("offline mode: required complete public cache is missing")
    last = None
    for attempt in range(4):
        try:
            headers = {"User-Agent": "bioinfo-coursework/2.0"}
            if data is not None:
                headers["Content-Type"] = "application/json"
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers),
                                        timeout=timeout) as r:
                result = json.load(r)
            if isinstance(result, dict) and result.get("errors"):
                raise RuntimeError("GraphQL errors: " + str(result["errors"])[:800])
            return result
        except (OSError, ValueError, RuntimeError) as exc:
            last = exc
            if attempt < 3:
                time.sleep(min(2 ** attempt, 4))
    raise RuntimeError("request failed: %s: %s" % (url, last)) from last


def graphql(query):
    return http_json(ENDPOINT, json.dumps({"query": query}).encode())["data"]


def resolve_targets(force=False):
    """Resolve every candidate explicitly; permit only documented symbol aliases."""
    cache = os.path.join(BENCH_DIR, "target_mapping.json")
    if os.path.isfile(cache) and not force:
        d = read_json(cache)
        if d.get("genes") == config.ALL_GENES:
            return d["mapping"]
    mapping = {}
    aliases = config.GENE_SYMBOL_ALIASES
    for start in range(0, len(config.ALL_GENES), 15):
        batch = config.ALL_GENES[start:start+15]
        parts = ['g%d: search(queryString:%s,entityNames:["target"],page:{index:0,size:10}) '
                 '{ hits { id name entity } }' % (i, json.dumps(g)) for i, g in enumerate(batch)]
        d = graphql("{" + " ".join(parts) + "}")
        for i, g in enumerate(batch):
            hits = [h for h in d["g%d" % i]["hits"] if h["name"] in [g, aliases.get(g)]]
            if len(hits) != 1:
                raise RuntimeError("ambiguous/unresolved target symbol: " + g)
            mapping[g] = dict(id=hits[0]["id"], approved_symbol=hits[0]["name"])
    write_json(cache, dict(fetch_date=today(), genes=config.ALL_GENES, aliases=aliases, mapping=mapping))
    return mapping


def fetch_open_targets(disease_key, force=False, mapping=None):
    """Query ALL resolved candidate IDs directly, avoiding global top-N truncation.

    Completeness is asserted for the fixed candidate universe, not all human genes.
    The filtered count must match the returned rows and all IDs must be requested.
    """
    ensure_dir(BENCH_DIR)
    cache = os.path.join(BENCH_DIR, "open_targets_%s.json" % disease_key)
    if os.path.exists(cache) and not force:
        old = read_json(cache)
        if old.get("schema_version") == 3 and old.get("complete") and old.get("universe") == config.ALL_GENES:
            if len(old["associations"]) == old["candidate_associated"]:
                return old
    mapping = mapping or resolve_targets()
    mondo = DISEASE_MONDO[disease_key]
    ids = [mapping[g]["id"] for g in config.ALL_GENES]
    query = ('{ disease(efoId:"%s") { id name '
             'allTargets: associatedTargets(page:{index:0,size:1}) { count } '
             'associatedTargets(Bs:%s,page:{index:0,size:1000}) '
             '{ count rows { target { id approvedSymbol } score } } } '
             'meta { apiVersion { x y z } dataVersion { year month } } }' % (mondo, json.dumps(ids)))
    d = graphql(query)
    disease = d["disease"]
    page = disease["associatedTargets"]
    by_id = {v["id"]: g for g, v in mapping.items()}
    rows = [dict(target_id=r["target"]["id"], symbol=by_id[r["target"]["id"]],
                 approved_symbol=r["target"]["approvedSymbol"], score=float(r["score"]))
            for r in page["rows"]]
    if len(rows) != page["count"] or len({r["target_id"] for r in rows}) != len(rows):
        raise RuntimeError("incomplete or duplicate candidate-universe query")
    out = dict(schema_version=3, complete=True, coverage_scope="all resolved candidate IDs",
               universe=config.ALL_GENES, disease_key=disease_key, mondo=mondo,
               name=disease["name"], fetch_date=today(), endpoint=ENDPOINT, metadata=d["meta"],
               total_associated=disease["allTargets"]["count"], candidate_associated=page["count"],
               pages=1, associations=rows)
    write_json(cache, out)
    print("[benchmark] %s: %d associated / %d resolved candidates" %
          (disease_key, len(rows), len(ids)), flush=True)
    return out


def fetch_drug_targets(force=False):
    cache = os.path.join(BENCH_DIR, "opentargets_drug_targets.json")
    if os.path.exists(cache) and not force:
        saved = read_json(cache)
        if set(saved) == set(config.DRUGS) and all("error" not in d for d in saved.values()):
            return saved
    out = {}
    for key in config.DRUGS:
        mid = None
        for field in ["pref_name__iexact", "molecule_synonyms__molecule_synonym__iexact"]:
            url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json?%s=%s&limit=3" % (
                field, urllib.parse.quote(key))
            mols = http_json(url).get("molecules", [])
            if mols:
                mid = mols[0]["molecule_chembl_id"]
                break
        if not mid:
            out[key] = dict(error="no ChEMBL molecule", targets=[], fetch_date=today())
            continue
        q = ('{ drug(chemblId:"%s") { name mechanismsOfAction { rows '
             '{ actionType mechanismOfAction targetName targets { id approvedSymbol } } } } }' % mid)
        drug = graphql(q)["drug"]
        if drug is None:
            out[key] = dict(error="no Open Targets drug", targets=[], fetch_date=today())
            continue
        targets = []
        for m in (drug.get("mechanismsOfAction") or {}).get("rows", []):
            for t in m.get("targets") or []:
                targets.append(dict(symbol=t["approvedSymbol"], target_id=t["id"],
                                    action_type=m.get("actionType"), moa=m.get("mechanismOfAction"),
                                    target_name=m.get("targetName"),
                                    complex_mapping=len(m.get("targets") or []) > 1))
        out[key] = dict(chembl_id=mid, pref_name=drug["name"], targets=targets, fetch_date=today())
        print("[benchmark] drug %s: %d links" % (key, len(targets)), flush=True)
    write_json(cache, out)
    return out


def fetch_string_network(genes, force=False):
    cache = os.path.join(BENCH_DIR, "string_network.json")
    if os.path.exists(cache) and not force:
        saved = read_json(cache)
        if saved.get("genes") == genes:
            return saved
    # Encode line breaks once, rather than double-encoding the literal '%0d'.
    url = "https://string-db.org/api/json/network?" + urllib.parse.urlencode(
        dict(identifiers="\r".join(genes), species=9606, required_score=400,
             caller_identity="bioinfo-coursework"))
    d = http_json(url, timeout=60)
    edges = [dict(a=e["preferredName_A"], b=e["preferredName_B"], score=float(e["score"]))
             for e in d if e.get("preferredName_A") in genes and e.get("preferredName_B") in genes]
    if not edges:
        raise RuntimeError("STRING returned no mapped edges")
    out = dict(fetch_date=today(), species=9606, genes=genes, edges=edges,
               network_type="functional association, not exclusively physical binding")
    write_json(cache, out)
    return out


def network_scorer(order, edges):
    ix = {g: i for i, g in enumerate(order)}
    A = np.zeros((len(order), len(order)))
    for e in edges:
        if e["a"] in ix and e["b"] in ix:
            i, j = ix[e["a"]], ix[e["b"]]
            A[i, j] = A[j, i] = max(A[i, j], e["score"])
    np.fill_diagonal(A, 0)
    return lambda seeds: A[:, seeds].sum(1)


def run_public_benchmark(representations, query_embs, gene_info, tfidf, universe, force=False):
    ensure_dir(BENCH_DIR)
    # Requests for independent diseases may run concurrently; each cache is written atomically.
    mapping = resolve_targets(force)
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {k: pool.submit(fetch_open_targets, k, force, mapping) for k in config.DISEASES}
        ot = {k: f.result() for k, f in futures.items()}
    drugs, string = fetch_drug_targets(force), fetch_string_network(universe, force)
    scorer = network_scorer(universe, string["edges"])
    rows2, rows3, sensitivity, details2, details3, labels = [], [], [], {}, {}, []
    for key, d in config.DISEASES.items():
        score_map = {r["symbol"]: r["score"] for r in ot[key]["associations"]}
        for g in universe:
            labels.append(dict(disease=key, gene=g, score=score_map.get(g, 0.),
                               has_record=g in score_map,
                               positive=score_map.get(g, 0.) >= OT_SCORE_THRESHOLD,
                               background_status="unlabelled_or_below_threshold"))
        for threshold in config.OT_THRESHOLDS:
            pos = [g for g in universe if score_map.get(g, 0.) >= threshold]
            rr, dd = evaluate_entity(key, d, "genes", representations, query_embs, tfidf,
                                     universe, "disease", pos, scorer)
            if not rr:
                sensitivity.append(dict(disease=key, threshold=threshold, n_pos=len(pos),
                                        note=dd["note"]))
            for r in rr:
                r.update(disease=key, disease_label=d["label"], threshold=threshold,
                         annotation_complete=True, source_fetch_date=ot[key]["fetch_date"])
                sensitivity.append(r)
                if threshold == OT_SCORE_THRESHOLD:
                    rows2.append(r)
            if threshold == OT_SCORE_THRESHOLD:
                details2[key] = dd
    for key, d in config.DRUGS.items():
        entry = drugs.get(key, {})
        aliases = {v: k for k, v in config.GENE_SYMBOL_ALIASES.items()}
        pos = sorted({aliases.get(t["symbol"], t["symbol"]) for t in entry.get("targets", [])
                      if aliases.get(t["symbol"], t["symbol"]) in universe})
        rr, dd = evaluate_entity(key, d, "targets", representations, query_embs, tfidf,
                                 universe, "drug", pos, scorer)
        details3[key] = dict(**dd, source=entry)
        for r in rr:
            r.update(drug=key, drug_label=d["label"], n_targets=len(pos),
                     source_fetch_date=entry.get("fetch_date", "unknown"))
            rows3.append(r)
        if not rr:
            rows3.append(dict(drug=key, drug_label=d["label"], n_targets=len(pos), note=dd["note"]))
    for name, rows in [("task2_public_benchmark", rows2), ("task3_public_benchmark", rows3),
                       ("task2_threshold_sensitivity", sensitivity), ("public_disease_labels", labels)]:
        pd.DataFrame(rows).to_csv(os.path.join(config.RESULTS_DIR, name + ".csv"),
                                 index=False, encoding="utf-8-sig")
    write_json(os.path.join(config.RESULTS_DIR, "task2_public_detail.json"), details2)
    write_json(os.path.join(config.RESULTS_DIR, "task3_public_detail.json"), details3)
    manifest = dict(evaluation_date=today(), threshold=OT_SCORE_THRESHOLD,
                    thresholds=list(config.OT_THRESHOLDS), universe=universe,
                    background_status="unlabelled, not experimentally verified negatives",
                    diseases={k: {f: v[f] for f in ["name", "mondo", "fetch_date", "metadata",
                                                    "total_associated", "pages", "complete"]}
                              for k, v in ot.items()},
                    drug_fetch_dates={k: v.get("fetch_date", "unknown") for k, v in drugs.items()},
                    string_fetch_date=string["fetch_date"], string_edges=len(string["edges"]))
    write_json(os.path.join(BENCH_DIR, "manifest.json"), manifest)
    return pd.DataFrame(rows2), pd.DataFrame(rows3)
