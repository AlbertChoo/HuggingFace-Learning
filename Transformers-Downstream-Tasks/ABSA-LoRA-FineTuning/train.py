import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
from data_preprocessing import tokenize_dataset, load_data

# Load and tokenize dataset
dataset = load_data("Laptop_Train_v2.xml", "Restaurants_Train_v2.xml")
tokenized_dataset = tokenize_dataset(dataset)

# Initialize Model with LoRA configuration
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["query", "key", "value", "dense"],
    lora_dropout=0.1, bias="none", task_type="SEQ_CLS"
)
model = get_peft_model(model, lora_config)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    optim="adamw_torch",
    save_steps=200,
    logging_steps=1,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine"
)

# Trainer Initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=None,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("./results/model")
    print("Model training completed.")
