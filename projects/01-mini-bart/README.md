# Mini-BART

## Working research question
When bart is compressed through knowledge distillation, does the smaller model preserve genuinely abstractive summarisation ability,or does it increasingly rely on extractive shortcuts?
[Does architectural compression and knowledge distillation preserve BART's abstractive summarization behavior, or does it change the balance between novel wording and source copying?]

## Planned comparisons
1. Fine-tuned BART teacher (reference).
2. Smaller BART student trained with supervised summarization loss (CE baseline).
3. Same student architecture trained with teacher–student distillation.

## Initial teacher
`facebook/bart-large-cnn` — pretrained and fine-tuned on CNN/DailyMail summarization.

## Notes
We will first run a tiny smoke test, then establish a reproducible baseline before distillation.
