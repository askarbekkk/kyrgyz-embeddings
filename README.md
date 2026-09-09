# Kyrgyz-E5: Adapting Multilingual Embeddings for Kyrgyz Semantic Retrieval

Fine-tuning multilingual embedding models for semantic search and RAG in Kyrgyz — a low-resource Turkic language with roughly 5 million speakers.

## Motivation

Kyrgyz NLP resources on Hugging Face cover speech recognition, text-to-speech, and translated knowledge benchmarks, but no widely evaluated embedding model optimized for Kyrgyz semantic retrieval was identified. This project measures how well existing multilingual models handle Kyrgyz retrieval, whether they beat a classical lexical baseline, and how much domain adaptation helps.

## Data

- **Source:** [Cleaned-Kyrgyz_Wikipedia](https://huggingface.co/datasets/Zhantas/Cleaned-Kyrgyz_Wikipedia) (CC-BY-SA-4.0), 76,519 articles
- **Chunking:** paragraphs of 200–800 characters → 14,250 candidate passages; residual MediaWiki markup stripped
- **Query generation:** an LLM produced one search-style Kyrgyz question per passage, with automatic failover across providers as free-tier quotas ran out
- **Filtering:** bibliography and citation fragments removed heuristically; remaining pairs re-checked by an LLM for answerability and language correctness
- **Final dataset:** 1,698 unique query–passage pairs, split 1,528 train / 170 test

## Evaluation

Retrieval over a 5,170-document corpus (5,000 distractor passages plus the 170 gold passages), one relevant document per query, measured with `InformationRetrievalEvaluator`.

An earlier setup scored every model at a perfect 1.0 because the corpus contained only the gold passages, making retrieval trivial. Expanding it to 5,000 distractors produced meaningful numbers and is the setup used throughout.

## Results

| Method | Accuracy@1 | nDCG@10 |
|---|---|---|
| BM25 (lexical baseline) | 0.624 | 0.770 |
| multilingual-e5-small | 0.554 | 0.784 |
| multilingual-e5-base | 0.618 | 0.821 |
| e5-base + fine-tune (batch 16, 1 epoch) | 0.635 | 0.833 |
| e5-base + fine-tune (batch 64, 1 epoch) | 0.618 | 0.825 |
| **e5-base + fine-tune (batch 64, 3 epochs)** | **0.635–0.641** | **0.841–0.843** |

Training: `MultipleNegativesRankingLoss`, lr 2e-5, fp16.

## Findings

**An untuned multilingual model does not clearly beat lexical search on Kyrgyz.** BM25 achieves higher Accuracy@1 than `multilingual-e5-base` out of the box (0.624 vs 0.618), though notably lower nDCG@10 (0.770 vs 0.821) — it either hits exactly or misses entirely, while the neural model keeps the correct passage near the top even when it ranks it second or third. Domain adaptation is what puts the neural model ahead on both metrics.

**Base model capacity outweighs fine-tuning at this data scale.** Moving from `e5-small` to `e5-base` gained more (+6.4 pp Accuracy@1) than any fine-tuning configuration applied to either.

**More synthetic data did not translate into more gain.** Scaling training data six-fold, from 278 to 1,528 pairs, did not increase the improvement over baseline. All queries came from one model, one source, and one template — additional examples from the same distribution teach little that is new. Diversity, not volume, appears to be the binding constraint.

**Batch size interacts with step count.** `MultipleNegativesRankingLoss` draws negatives from within the batch, so a larger batch makes the task harder. At batch 64 with one epoch only 24 optimizer steps ran and training loss stayed at 0.39 — undertrained. Three epochs at the same batch size gave the best result.

**Overfitting boundary.** On the smaller 278-pair dataset, training past one epoch degraded performance below the untuned baseline (0.589 vs 0.607 Accuracy@1).

**Variance is comparable to the effect size.** With 170 test queries, one query equals 0.6 pp. Two runs of an identical configuration produced 0.635 and 0.641, so the table reports ranges rather than best-run numbers.

**Tokenization cost.** Cyrillic Kyrgyz passages averaged ~575 tokens per request, well above comparable English text — a consequence of multilingual tokenizers being fitted primarily to Latin-script, high-resource languages, and a direct cost multiplier for API-based work on Kyrgyz.

## Limitations

- Queries are LLM-generated and contain occasional grammatical errors, particularly in case endings; no systematic native-speaker validation of the full set was performed.
- The BM25 baseline uses naive whitespace tokenization with no stemming. Kyrgyz is agglutinative, so inflected forms are treated as distinct terms — BM25 numbers here are a lower bound.
- Wikipedia is the only source; the test set does not represent conversational, domain-specific, or code-switched queries.
- No hard negatives were mined. In-batch negatives are sampled at random and are therefore easy to distinguish, which likely caps the achievable gain.
- Single random seed per configuration; results are not averaged across runs.

## Next steps

Mine hard negatives from top-k retrieval results, evaluate `multilingual-e5-large` and BGE-M3, add stemming to the BM25 baseline, diversify query types beyond factoid questions, and average results across several seeds.

## Stack

Python, PyTorch, sentence-transformers, rank_bm25, Hugging Face Datasets and Hub, Google Colab (T4).

## License

CC-BY-SA-4.0, inherited from the source dataset.
