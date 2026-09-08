# Kyrgyz Embeddings

Fine-tuning multilingual embedding models for semantic search and RAG in Kyrgyz — a low-resource Turkic language spoken by ~5M people.

## Motivation

No embedding model exists for Kyrgyz retrieval. Existing Kyrgyz NLP work on Hugging Face covers ASR, TTS, and knowledge benchmarks, but nothing for semantic search — which makes RAG applications in Kyrgyz impractical. This project measures how well multilingual models handle Kyrgyz retrieval, and whether light fine-tuning helps.

## Data

- **Source:** [Cleaned-Kyrgyz_Wikipedia](https://huggingface.co/datasets/Zhantas/Cleaned-Kyrgyz_Wikipedia) (CC-BY-SA-4.0), 76,519 articles
- **Chunking:** paragraphs of 200–800 characters → 14,250 passages
- **Query generation:** an LLM generated one natural search-style question per passage
- **Filtering:** each pair was re-checked by an LLM for answerability; 278 of 300 retained
- **Split:** 222 train / 56 test

## Evaluation

Retrieval over a corpus of 5,056 passages with 56 test queries, one relevant document each. Metrics via `InformationRetrievalEvaluator`.

| Model | Accuracy@1 | nDCG@10 |
|---|---|---|
| multilingual-e5-small | 0.554 | 0.784 |
| multilingual-e5-small + fine-tune | 0.571 | 0.810 |
| multilingual-e5-base | 0.607 | 0.834 |
| **multilingual-e5-base + fine-tune** | **0.643** | **0.847** |

Training: `MultipleNegativesRankingLoss`, 1 epoch, batch size 16, lr 2e-5.

## Findings

- Model size matters more than fine-tuning at this data scale: switching from `small` to `base` gave a larger gain (+5.3 pp accuracy@1) than fine-tuning either model.
- More than one epoch overfits. At 2 epochs, accuracy@1 dropped to 0.589 — below the un-tuned baseline.
- **Variance caveat:** with only 56 test queries, one query equals 1.8 pp. Two runs of the identical 1-epoch configuration produced 0.625 and 0.643, so differences under ~4 pp should not be treated as meaningful.

## Limitations

- 278 training pairs is small; results are preliminary.
- Queries are LLM-generated, not written by native speakers.
- Test set is Wikipedia-only and does not reflect conversational or domain-specific queries.
- No hard negatives were mined.

## Next steps

Scale the dataset to 2,000+ pairs, add native-speaker validation, mine hard negatives, and evaluate BGE-M3.

## License

CC-BY-SA-4.0, inherited from the source dataset.
