"""Parameter-free full-sequence composition and corpus-only TF-IDF baselines."""
import re
from collections import Counter
import numpy as np


class Tfidf:
    def __init__(self, texts):
        docs = [self.tokens(t) for t in texts]
        self.vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
        df = np.zeros(len(self.vocab))
        for d in docs:
            for w in set(d):
                df[self.vocab[w]] += 1
        self.idf = np.log((1 + len(docs)) / (1 + df)) + 1
        self.X = self.transform(texts)

    @staticmethod
    def tokens(text):
        text = re.sub(r"\{[^}]*\}|PubMed:\s*\d+", " ", text, flags=re.I)
        return re.findall(r"[a-z][a-z0-9]*", text.lower())

    def transform(self, texts):
        X = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, t in enumerate(texts):
            for w, count in Counter(self.tokens(t)).items():
                if w in self.vocab:
                    X[i, self.vocab[w]] = (1 + np.log(count)) * self.idf[self.vocab[w]]
        return X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)


def build_representations(gene_embs, gene_info, seqs, genes):
    from .embeddings import gene_texts
    order = [g for g in genes if all(g in e for e in gene_embs.values()) and g in seqs]
    reps = {name: {g: embs[g] for g in order} for name, embs in gene_embs.items()}
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    reps["aac"] = {g: np.array([seqs[g].count(a) / max(len(seqs[g]), 1) for a in alphabet],
                               dtype=np.float32) for g in order}
    tfidf = Tfidf(gene_texts(gene_info, order))
    reps["tfidf"] = dict(zip(order, tfidf.X))
    if "esm" in reps and "biobert" in reps:
        reps["fusion"] = {g: np.concatenate([reps["esm"][g], reps["biobert"][g]]) / np.sqrt(2)
                          for g in order}
    return reps, tfidf, order
