# Kyrgyz semantic search demo

BM25 vs BGE-M3 side by side on human-written Kyrgyz questions (Belebele `kir_Cyrl`).

![demo](demo_search.png)

**Corpus:** 488 Belebele (FLORES) passages + 200 Kyrgyz Wikipedia distractors. **All 900 questions scored:**

| Method | Accuracy@1 |
|---|---|
| BM25 | 0.570 |
| BGE-M3 (zero-shot) | **0.841** |

This laptop-sized corpus is smaller than the research setup (5,000 distractors: BM25 0.484, BGE-M3 0.794), so absolute numbers are higher; the gap is the same story.

```bash
ollama pull bge-m3
pip install -r requirements.txt
python build_index.py      # embeds the corpus and questions, writes index.npz / corpus.json / metrics.json
streamlit run app.py
```

Data: Belebele (CC BY-SA 4.0), Cleaned-Kyrgyz_Wikipedia (CC BY-SA 4.0).
