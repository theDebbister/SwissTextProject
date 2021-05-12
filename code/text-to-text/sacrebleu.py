import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
import datasets
from datasets import load_dataset

result_path = "translation1.out"
refs = "test_dataset.txt"
refs2 = "train_dataset.txt"

results = pd.read_csv(result_path, names=["res"])
pd_refs1 = pd.read_csv(refs, sep = ";", names=["src","trg"])
pd_refs2 = pd.read_csv(refs2, sep = ";", names=["src","trg"])

refs = pd.concat([pd_refs1, pd_refs2])

print(pd_refs1.head())
print(pd_refs2.head())

print(len(refs))
print(len(results))

train_path = 'train_dataset.txt'
test_path = 'test_dataset.txt'

preds = results['res'].to_numpy()
refs = refs['trg'].to_numpy()

def calculate_corpus_bleu(refs, preds):
    references = [[sample] for sample in refs]
    predictions = [sample for sample in preds]

    bleu_score = corpus_bleu(references, predictions)

    return bleu_score

bleu = calculate_corpus_bleu(refs, preds)
print(bleu)
