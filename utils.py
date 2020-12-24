def process_dataset(dataset,tokenizer):
    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)
    dataset = dataset.map(tokenize, batched=True, batch_size=len(dataset))
    if 'label' in dataset.column_names:
        dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    else:
        dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
    return dataset
