# BabyLM Performance Comparison Report

## Method

This report compares two small language models (BabyLMs) trained from scratch. Both models use the LLaMA architecture, but they differ in size and in the amount of data.

- Model 1 (Small): 2 MB of training text, 64 hidden units, 2 layers, 2 attention heads, 8K vocabulary
- Model 2 (Large): 4 MB of training text, 96 hidden units, 3 layers, 3 attention heads, 10K vocabulary

We trained both models on Wikitext‑2 for 50 epochs. We used the same settings: learning rate 0.0003, batch size 32, and a cosine learning rate schedule. We evaluated the models on 1,000 minimal pairs from the BLiMP dataset (English). These pairs test several linguistic phenomena, such as agreement and morphology. The decision rule is simple: the model is correct if the grammatical sentence has lower perplexity than the ungrammatical one.

## Results

### Overall accuracy
- Model 1 (Small): 53.90%
- Model 2 (Large): 50.40%

The smaller model performed better overall, even though it used less data. This shows that a larger model is not always better on small training sets.

### Accuracy by phenomenon (examples)

| Phenomenon | Model 1 (Small) | Model 2 (Large) | Difference |
|------------|-----------------|-----------------|------------|
| Anaphor Gender Agreement | 60.00% | 50.00% | +10.00% |
| Determiner‑Noun Agreement 1 | 50.00% | 56.82% | −6.82% |
| Determiner‑Noun Agreement 2 | 50.00% | 50.00% | 0.00% |
| Determiner‑Noun Agreement 3 | 50.00% | 50.00% | 0.00% |
| Determiner‑Noun Agreement 4 | 50.00% | 50.00% | 0.00% |
| Irregular Past | 50.00% | 50.00% | 0.00% |
| Regular Past | 50.00% | 50.00% | 0.00% |
| Subject‑Verb Agreement 1 | 50.00% | 50.00% | 0.00% |
| Subject‑Verb Agreement 2 | 50.00% | 50.00% | 0.00% |
| Subject‑Verb Agreement 3 | 50.00% | 50.00% | 0.00% |

### Main observations
1. Model 1 is stronger on anaphor gender agreement (60% vs 50%). This suggests better handling of pronoun–antecedent links.
2. Model 2 is slightly better on determiner–noun agreement (56.82% vs 50%). This may show a small gain in morphology.
3. Many categories are near 50%, which is the chance level. Both models still struggle to generalize across phenomena.
4. Training time: Model 1 trained in about 13 minutes. Model 2 took about 56 minutes. This is a big time difference for similar results.

## How the differences affected performance

- Model size and data amount: The larger model trained on 4 MB was not better than the smaller model on 2 MB (50.40% vs 53.90%). With very little data, extra parameters are hard to train and may not learn stable patterns. The smaller model seems to learn more stable rules in this setting.
- Tokenizer and vocabulary: A 10K vocabulary (Model 2) did not help over 8K (Model 1). With short context and small data, the gain from a larger vocab is limited.
- Context length: Short context (≤96 tokens) restricts the types of dependencies the models can learn. This likely kept many phenomena near chance level for both models.
- Phenomenon‑level effects: The small model was better on anaphor gender agreement, while the large one was slightly better on determiner–noun agreement. This suggests each setting picks up different cues, but neither generalizes strongly.
- Efficiency and cost: The large model took about 4× longer to train for similar accuracy. In a low‑resource setting, the small model is a better choice for time and resources.

## My Contribution

I designed and ran the full pipeline under low resources.

- I prepared training data: 2 MB and 4 MB subsets from Wikitext‑2.
- I trained a custom byte‑level BPE tokenizer and saved it for both models.
- I set up two LLaMA configs (Small vs Large) and kept other settings the same to make a fair comparison.
- I trained both models from scratch, enabled Apple MPS for speed, and disabled external logging.
- I built the evaluation: 1,000 BLiMP minimal pairs (English) and used perplexity to decide between grammatical and ungrammatical sentences.
- I generated training‑loss plots and a comparison figure by phenomenon.

### Impact of my choices on results

- Choosing different model sizes and data amounts let me compare options under low resources. My small model on 2 MB was better than the large one on 4 MB (53.90% vs 50.40%), so making the model bigger did not help in this setup.
- Keeping most training settings the same made the comparison fair. This shows the accuracy gap is mainly from model size and data, not from other hyperparameters.
- Using an 8K vs 10K vocabulary did not change results much. This suggests that with short context and little data, a bigger vocab brings little benefit.
- The evaluation design (perplexity on minimal pairs) revealed where each model is better. My small model handled anaphor gender agreement better, while the large one was slightly better on determiner–noun agreement.
- Focusing on Apple MPS and simple logging kept training efficient. The small model finished ~4× faster with similar or better accuracy, which fits the low‑resource goal.

## Discussion

In this low‑resource setting, a bigger model did not help. With only a few megabytes of text, the larger model did not learn more useful patterns. The smaller model may regularize better and focus on simpler rules. Also, our context length and vocabulary sizes are small, which limits the ceiling for both models.

## Conclusion

Under very small data conditions, simpler BabyLMs can match or beat larger ones. Careful choices in model size and training time can be more important than adding more parameters. In future work, we could test more data, longer context windows, or improved tokenization to see if the larger model starts to win.
