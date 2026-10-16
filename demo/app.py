"""Kyrgyz semantic search demo: BM25 vs BGE-M3 side by side.

    python build_index.py          # once
    streamlit run app.py
"""
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi

HERE = Path(__file__).parent
st.set_page_config(page_title="Kyrgyz Semantic Search", layout="wide")


@st.cache_resource
def load():
    data = json.loads((HERE / "corpus.json").read_text(encoding="utf-8"))
    docs = np.load(HERE / "index.npz")["docs"]
    bm25 = BM25Okapi([tokenize(d) for d in data["corpus"]])
    return data["corpus"], data["questions"], docs, bm25, json.loads((HERE / "metrics.json").read_text())


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def embed(text):
    req = urllib.request.Request("http://localhost:11434/api/embed", json.dumps({"model": "bge-m3", "input": [text]}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        v = np.array(json.load(r)["embeddings"][0], dtype=np.float32)
    return v / np.linalg.norm(v)


corpus, questions, docs, bm25, metrics = load()

st.title("Kyrgyz Semantic Search · Кыргызча семантикалык издөө")
st.caption(f"Corpus: {metrics['n_passages']} Belebele (FLORES) passages + {metrics['n_corpus'] - metrics['n_passages']} "
           "Kyrgyz Wikipedia distractors · questions are human-written (Belebele kir_Cyrl)")

c1, c2, c3 = st.columns(3)
c1.metric("BM25 · Accuracy@1", f"{metrics['BM25']:.3f}")
c2.metric("BGE-M3 · Accuracy@1", f"{metrics['BGE-M3']:.3f}", f"{(metrics['BGE-M3'] - metrics['BM25']) * 100:+.1f} pp")
c3.metric("Questions scored", metrics["n_questions"])

idx = st.selectbox("Pick a Belebele question (or type your own below)", range(len(questions)),
                   format_func=lambda i: questions[i]["q"][:140], index=3)
query = st.text_input("Query", value=questions[idx]["q"])
gold = questions[idx]["gold"] if query == questions[idx]["q"] else None

bm_scores = bm25.get_scores(tokenize(query))
dense_scores = docs @ embed(query)


def show(col, title, scores):
    col.subheader(title)
    for rank, i in enumerate(np.argsort(-scores)[:5], 1):
        mark = "✅ " if i == gold else ""
        box = col.container(border=True)
        box.markdown(f"**{mark}#{rank}** · score {scores[i]:.3f}" + (" · *correct passage*" if i == gold else ""))
        box.write(corpus[i][:420] + ("…" if len(corpus[i]) > 420 else ""))


left, right = st.columns(2)
show(left, "BM25 (keyword match)", bm_scores)
show(right, "BGE-M3 (dense embeddings)", dense_scores)
