import numpy as np
import random
import pandas as pd
from random import randint, seed
import json
import datasets
from datasets import load_dataset, load_metric
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from IPython.display import display, HTML

MODEL_SAVE_PATH = "./model/fineTuned3"

# LOAD DATA
with open('../data/text2.json') as f:
    data = json.load(f)

def build_text_files(data_json, dest_path):
    f = open(dest_path, 'w')
    data = ''
    for texts in data_json:
        trg = str(texts['src']).strip()
        src = str(texts['res']).strip()
        if src and trg:
            summary = src + ";" + trg + "\n"
            f.write(summary)
    f.close()

randValue = randint(1, 100)
train, test = train_test_split(data,test_size=0.15, shuffle=True, random_state=randValue)

build_text_files(train,'train_dataset.txt')
build_text_files(test,'test_dataset.txt')

print("Train dataset length: "+str(len(train)))
print("Test dataset length: "+ str(len(test)))

train_path = 'train_dataset.txt'
test_path = 'test_dataset.txt'

model_checkpoint = "Helsinki-NLP/opus-mt-de-en"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def load_datasets(train_path,test_path):
    return load_dataset('csv', data_files={'train': train_path, 'test': test_path}, delimiter=';', column_names = ['src','trg'])

datasets= load_datasets(train_path, test_path)
metric = load_metric("sacrebleu")
fake_preds = ["hello there", "general kenobi"]
fake_labels = [["hello there"], ["general kenobi"]]
metric.compute(predictions=fake_preds, references=fake_labels)

if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "translate English to Romanian: "
else:
    prefix = ""

max_input_length = 128
max_target_length = 128
source_lang = "src"
target_lang = "trg"

def preprocess_function(examples):
    model_inputs = tokenizer(examples['src'], max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["trg"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = datasets.map(preprocess_function, batched=True, batch_size=500)

######
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
batch_size = 24
args = Seq2SeqTrainingArguments(
    "test-translation",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

###

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
model.save_pretrained(MODEL_SAVE_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_SAVE_PATH)
print(model)
