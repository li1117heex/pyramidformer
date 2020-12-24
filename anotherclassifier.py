import os, torch

from datasets import load_dataset, concatenate_datasets, load_from_disk

from transformers import FunnelTokenizer, FunnelForMaskedLM, Trainer, DataCollatorForLanguageModeling, \
    FunnelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from config import modelconfig, training_args_pt, training_args_ft, prefix

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# dataset_path = "bookcorpus+wikipedia_shuffle"
# dataset_path = ["bookcorpus0", "bookcorpus1", "wikipedia"]
# cached_datasets_path = prefix + "cached_datasets/"

tokenizer = FunnelTokenizer.from_pretrained('funnel-transformer/small')

def tokenize(batch):
    return tokenizer(batch['sentence'], padding='max_length', truncation=True, max_length=512)

dataset = load_dataset('glue','sst2')
# dataset = concatenate_datasets([load_from_disk(cached_datasets_path + path) for path in dataset_path]).shuffle()

# dataset1 = load_from_disk(cached_datasets_path + dataset_path)

# dataset1 = load_dataset("wikipedia", name='20200501.en', split='train')
# dataset1.remove_columns_('title')
# dataset2 = load_dataset("bookcorpus", split='train')
# dataset = concatenate_datasets([dataset1, dataset2]).shuffle()
#
# dataset = dataset.map(tokenize, batched=True, batch_size=512, num_proc=32)
# dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
# dataset.save_to_disk(cached_datasets_path+dataset_path)

# dataset = load_dataset('glue','sst2')['test']
dataset=dataset.map(tokenize, batched=True, batch_size=512, num_proc=1)
# dataset.remove_columns_(['label', 'idx'])
# # dataset.set_format('torch', columns=['input_ids', 'attention_mask'])


model = FunnelForSequenceClassification(config=modelconfig)

trainer = Trainer(
    model=model,
    args=training_args_pt,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics
)
trainer.train()
trainer.evaluate()
result = trainer.predict(dataset['validation'])

model.save_pretrained(training_args_pt.output_dir)
torch.save(model, training_args_pt.output_dir+"torch")



# train_dataset, test_dataset = load_from_disk(cached_datasets_path+dataset_path2, split=['train', 'test'])





model2.save_pretrained(training_args_ft.output_dir)
torch.save(model2, training_args_ft.output_dir+"torch")

