import json
import random
import pandas as pd
import datasets
from datasets import load_dataset, load_metric
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
from IPython.display import display, HTML

train_path = 'train_dataset.txt'
test_path = 'test_dataset.txt'

model_checkpoint = "./model/fineTuned3"

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en')
def load_datasets(train_path,test_path):
    return load_dataset('csv', data_files={'train': [train_path, test_path]}, delimiter=';', column_names = ['src','trg'])

datasets= load_datasets(train_path, test_path)

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

# Initialize the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Tokenize text
def translate(text):
    tokenized_text = tokenizer.prepare_seq2seq_batch(text['src'], return_tensors='pt')
    translation = model.generate(**tokenized_text)
    print(tokenizer.batch_decode(translation, skip_special_tokens=True)[0])

datasets['train'].map(translate)
