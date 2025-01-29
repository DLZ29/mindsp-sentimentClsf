import os

import mindspore
from mindspore.dataset import text, GeneratorDataset, transforms
from mindspore import nn

from mindnlp.dataset import load_dataset
from mindnlp.engine import Trainer

imdb_ds = load_dataset('imdb', split=['train', 'test'])
imdb_train = imdb_ds['train']
imdb_test = imdb_ds['test']

imdb_train.get_dataset_size()

import numpy as np


def process_dataset(dataset, tokenizer, max_seq_len=512, batch_size=4, shuffle=False):
    is_ascend = mindspore.get_context('device_target') == 'Ascend'

    def tokenize(text):
        if is_ascend:
            tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=max_seq_len)
        else:
            tokenized = tokenizer(text, truncation=True, max_length=max_seq_len)
        return tokenized['input_ids'], tokenized['attention_mask']

    if shuffle:
        dataset = dataset.shuffle(batch_size)

    dataset = dataset.map(operations=[tokenize], input_columns="text", output_columns=['input_ids', 'attention_mask'])
    dataset = dataset.map(operations=transforms.TypeCast(mindspore.int32), input_columns="label",
                          output_columns="labels")

    if is_ascend:
        dataset = dataset.batch(batch_size)
    else:
        dataset = dataset.padded_batch(batch_size, pad_info={'input_ids': (None, tokenizer.pad_token_id),
                                                             'attention_mask': (None, 0)})

    return dataset


from mindnlp.transformers import OpenAIGPTTokenizer

gpt_tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

special_tokens_dict = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
}
num_added_toks = gpt_tokenizer.add_special_tokens(special_tokens_dict)

imdb_train, imdb_val = imdb_train.split([0.7, 0.3])

dataset_train = process_dataset(imdb_train, gpt_tokenizer, shuffle=True)
dataset_val = process_dataset(imdb_val, gpt_tokenizer)
dataset_test = process_dataset(imdb_test, gpt_tokenizer)

next(dataset_train.create_tuple_iterator())

from mindnlp.transformers import OpenAIGPTForSequenceClassification

model = OpenAIGPTForSequenceClassification.from_pretrained('openai-gpt', num_labels=2)
model.config.pad_token_id = gpt_tokenizer.pad_token_id
model.resize_token_embeddings(model.config.vocab_size + 3)

from mindnlp.engine import TrainingArguments

training_args = TrainingArguments(
    output_dir="gpt_imdb_finetune",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    num_train_epochs=1.0
)

from mindnlp import evaluate
import numpy as np

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    compute_metrics=compute_metrics
)

trainer.train()

import pandas as pd


def load_csv_dataset(file_path):
    data = pd.read_csv(file_path)
    return data['text'], data['label']


def process_twitter(texts, labels, tokenizer, max_seq_len=512, batch_size=4):
    is_ascend = mindspore.get_context('device_target') == 'Ascend'

    def tokenize(text):
        tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=max_seq_len)
        return tokenized['input_ids'], tokenized['attention_mask']

    def generator():
        for text, label in zip(texts, labels):
            input_ids, attention_mask = tokenize(text)
            yield np.array(input_ids), np.array(attention_mask), np.array(label, dtype=np.int32)

    dataset = GeneratorDataset(generator, column_names=["input_ids", "attention_mask", "labels"])

    if is_ascend:
        dataset = dataset.batch(batch_size)
    else:
        dataset = dataset.padded_batch(batch_size, pad_info={'input_ids': (None, tokenizer.pad_token_id),
                                                             'attention_mask': (None, 0)})

    return dataset


ft_train_texts, ft_train_labels = load_csv_dataset('train50.csv')
ft_val_texts, ft_val_labels = load_csv_dataset('test500.csv')
ft_test_texts, ft_test_labels = load_csv_dataset('test500.csv')

ft_dataset_train = process_twitter(ft_train_texts, ft_train_labels, gpt_tokenizer)
ft_dataset_val = process_twitter(ft_val_texts, ft_val_labels, gpt_tokenizer)
ft_dataset_test = process_twitter(ft_test_texts, ft_test_labels, gpt_tokenizer)

finetuning_args = TrainingArguments(
    output_dir="gpt_imdb_finetune",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    num_train_epochs=1.0
)

finetuner = Trainer(
    model=model,
    args=finetuning_args,
    train_dataset=ft_dataset_train,
    eval_dataset=ft_dataset_val,
    compute_metrics=compute_metrics
)

finetuner.train()

finetuner.evaluate(ft_dataset_test)
