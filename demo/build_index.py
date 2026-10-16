"""Build the demo search index: Belebele kir_Cyrl passages + Kyrgyz Wikipedia distractors.

Embeds everything with BGE-M3 through a local Ollama server and scores BM25 vs BGE-M3 on all
900 human-written Belebele questions. Output: index.npz + corpus.json + metrics.json.

    ollama pull bge-m3 && python build_index.py
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

HERE = Path(__file__).parent
N_DISTRACTORS = 200  # laptop CPU: ~8 passages/min with BGE-M3
OLLAMA = "http://localhost:11434/api/embed"


def hf_rows(dataset, config, split, total):
    cache = HERE / f"raw_{dataset.replace('/', '_')}_{config}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    rows = []
    for off in range(0, total, 100):
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": off, "length": 100})
        for attempt in range(5):  # the datasets-server API returns occasional 502s
            try:
                with urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{q}", timeout=60) as r:
                    rows += [x["row"] for x in json.load(r)["rows"]]
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 4:
                    raise
                time.sleep(3 * (attempt + 1))
    cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def embed(texts, batch=16):
    out = []
    for i in range(0, len(texts), batch):
        req = urllib.request.Request(OLLAMA, json.dumps({"model": "bge-m3", "input": texts[i:i + batch]}).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            out += json.load(r)["embeddings"]
        print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    v = np.array(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def tokenize(text):  # same naive whitespace tokenization as the research baseline
    return re.findall(r"\w+", text.lower())


def main():
    bele = hf_rows("facebook/belebele", "kir_Cyrl", "test", 900)
    passages = list(dict.fromkeys(r["flores_passage"] for r in bele))
    wiki = hf_rows("Zhantas/Cleaned-Kyrgyz_Wikipedia", "default", "train", 600)
    paras = [p.strip() for row in wiki for p in row["text"].split("\n") if 200 <= len(p.strip()) <= 800]
    corpus = passages + [p for p in dict.fromkeys(paras) if p not in passages][:N_DISTRACTORS]
    questions = [{"q": r["question"], "gold": passages.index(r["flores_passage"])} for r in bele]
    print(f"{len(passages)} Belebele passages + {len(corpus) - len(passages)} distractors, {len(questions)} questions")

    doc_vecs = embed(corpus)
    q_vecs = embed([x["q"] for x in questions])
    bm25 = BM25Okapi([tokenize(d) for d in corpus])

    hits = {"BM25": [], "BGE-M3": []}
    for x, qv in zip(questions, q_vecs):
        hits["BM25"].append(int(np.argmax(bm25.get_scores(tokenize(x["q"])))) == x["gold"])
        hits["BGE-M3"].append(int(np.argmax(doc_vecs @ qv)) == x["gold"])
    metrics = {k: round(float(np.mean(v)), 3) for k, v in hits.items()}
    metrics.update(n_questions=len(questions), n_passages=len(passages), n_corpus=len(corpus))
    print("Accuracy@1:", metrics)

    np.savez_compressed(HERE / "index.npz", docs=doc_vecs)
    (HERE / "corpus.json").write_text(json.dumps({"corpus": corpus, "questions": questions}, ensure_ascii=False), encoding="utf-8")
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
