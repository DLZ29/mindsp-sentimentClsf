import os
import pandas as pd
import mindspore
from mindspore.dataset import GeneratorDataset, transforms
from mindnlp.transformers import OpenAIGPTTokenizer
from mindnlp.engine import Trainer, TrainingArguments
from mindnlp import evaluate
import numpy as np


def load_csv_dataset(file_path):
    data = pd.read_csv(file_path)
    return data['text'], data['label']


def process_dataset(texts, labels, tokenizer, max_seq_len=512, batch_size=4):
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


train_texts, train_labels = load_csv_dataset('train2.csv')
val_texts, val_labels = load_csv_dataset('test3.csv')
test_texts, test_labels = load_csv_dataset('test3.csv')

print(train_texts[0:2])
print(train_labels[0:2])

gpt_tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

special_tokens_dict = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
}
num_added_toks = gpt_tokenizer.add_special_tokens(special_tokens_dict)

dataset_train = process_dataset(train_texts, train_labels, gpt_tokenizer)
dataset_val = process_dataset(val_texts, val_labels, gpt_tokenizer)
dataset_test = process_dataset(test_texts, test_labels, gpt_tokenizer)

next(dataset_train.create_tuple_iterator())

from mindnlp.transformers import OpenAIGPTForSequenceClassification

model = OpenAIGPTForSequenceClassification.from_pretrained('openai-gpt', num_labels=2)
model.config.pad_token_id = gpt_tokenizer.pad_token_id
model.resize_token_embeddings(model.config.vocab_size + 3)

training_args = TrainingArguments(
    output_dir="gpt_imdb_finetune",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    num_train_epochs=1.0
)

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

trainer.evaluate(dataset_test)




