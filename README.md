<img width="875" height="557" alt="3" src="https://github.com/user-attachments/assets/ffda2368-48d0-45a9-9b4e-2f5882b5994c" />


<img width="865" height="557" alt="4" src="https://github.com/user-attachments/assets/ff30c690-75e7-41f6-b29d-a93323870034" />
# Kyrgyz-E5: Adapting Multilingual Embeddings for Kyrgyz Semantic Retrieval

Fine-tuning multilingual embedding models for semantic search and RAG in Kyrgyz — a low-resource Turkic language with roughly 5 million speakers.

## Update: corrected evaluation + external benchmark

**Bug found in the original evaluation.** Distractors were drawn from the same passage pool the queries were generated from, so 58 of 170 gold passages also appeared in the corpus under a second ID. Retrieving that identical copy was scored as a miss, which deflated every method. After removing the duplicates:

| Method | Accuracy@1 | nDCG@10 |
|---|---|---|
| BM25 | 0.729 | 0.811 |
| multilingual-e5-base (untuned) | 0.877 | 0.921 |
| e5-base fine-tuned, random negatives | 0.906 | 0.944 |
| **e5-base fine-tuned, hard negatives** | **0.918** | **0.950** |

**External test: [Belebele](https://huggingface.co/datasets/facebook/belebele) (`kir_Cyrl`).** 900 human-written questions; the model must retrieve the right FLORES passage among 488 passages plus 5,000 Wikipedia distractors. No Belebele data was used in training.

| Method | Accuracy@1 | nDCG@10 |
|---|---|---|
| BM25 | 0.484 | 0.572 |
| multilingual-e5-base (untuned) | 0.559 | 0.671 |
| e5-base fine-tuned, random negatives | **0.740** | 0.808 |
| e5-base fine-tuned, hard negatives | 0.734 | **0.811** |

**Takeaways.** Fine-tuning on ~1.5k synthetic Wikipedia pairs gives **+18 pp Accuracy@1** on human-written questions from a different domain, and the untuned neural model already clearly beats BM25. Adding E5's `query:` / `passage:` prefixes did not help (untuned: 0.847 vs 0.877 Accuracy@1). Training used the unfiltered pairs; the LLM-filtered set has not been tried yet.

**Zero-shot comparison of multilingual encoders.** Same setup as above (fixed synthetic test and Belebele), Accuracy@1 / nDCG@10:

| Model | Synthetic (fixed) | Belebele |
|---|---|---|
| **BAAI/bge-m3** | **0.929 / 0.958** | **0.794 / 0.866** |
| e5-base fine-tuned, hard negatives (ours) | 0.918 / 0.950 | 0.734 / 0.810 |
| multilingual-e5-large | 0.888 / 0.929 | 0.552 / 0.688 |
| multilingual-e5-base | 0.876 / 0.921 | 0.560 / 0.671 |
| multilingual-e5-small | 0.829 / 0.892 | 0.432 / 0.560 |
| Qwen3-Embedding-0.6B | 0.794 / 0.867 | 0.543 / 0.634 |
| BM25 | 0.729 / 0.811 | 0.484 / 0.572 |
| LaBSE | 0.524 / 0.631 | 0.443 / 0.550 |

BGE-M3 without any Kyrgyz fine-tuning is the strongest model on both tests and beats our fine-tuned e5-base. Fine-tuning still closes most of the gap for e5-base, so fine-tuning BGE-M3 is the next step. Single runs; confidence intervals are not reported yet.

> The sections below were written before this fix. Their absolute numbers and the conclusions that depend on them (e.g. BM25 vs. untuned E5) are superseded by the tables above.

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

## Final Results (3 seeds)

| Method | Accuracy@1 | nDCG@10 |
|---|---|---|
| Baseline (no fine-tuning) | 0.618 | 0.821 |
| Fine-tuned, random negatives | 0.629 ± 0.008 | 0.838 ± 0.004 |
| **Fine-tuned, hard negatives** | **0.641 ± 0.005** | **0.845 ± 0.003** |

<img width="865" height="557" alt="4" src="https://github.com/user-attachments/assets/f2bff51b-fd4c-4eb4-9cbe-85efd16a4672" />


Each configuration was trained with 3 seeds (42, 43, 44) to separate real effects from initialization noise — a single-seed comparison earlier in this project showed a spread (0.635–0.659) larger than the apparent gain between methods. With multiple seeds, hard negatives show a consistent improvement over random negatives (+1.2 pp Accuracy@1) that exceeds the standard deviation of either method, and notably lower variance (±0.005 vs ±0.008), suggesting more stable training.

## Stack

Python, PyTorch, sentence-transformers, rank_bm25, Hugging Face Datasets and Hub, Google Colab (T4).

## License

CC-BY-SA-4.0, inherited from the source dataset.
