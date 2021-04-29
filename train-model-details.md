# Trained Models

## Baseline

- full swiss text data
- german text corpus kenlm
- learning rate 0.0005
- dropout 0.4
- epochs ca. 60
- n-hiddenstates 1024
- tested on test07.csv
- WER: 0.746646, CER: 0.532223, loss: 152.055420, BLEU: 0.100805

## Fine-Tune small
- lm from realease 0.9.0
- 5 epochs
- tested on test07
- no further parameter changes
- WER: 0.730518, CER: 0.511030, loss: 147.559677, BLEU: 0.101


