# Multilingual Named Entity Recognition (NER) with XLM-RoBERTa

## 🚀 Project Overview

This project focuses on Multilingual Named Entity Recognition (NER) using XLM-RoBERTa, a transformer-based model capable of handling multiple languages. The objective is to fine-tune the model to identify entities such as persons, organizations, and locations across different languages.

## 📂 Dataset

The dataset used for this project is WikiANN (Pan-X), a multilingual NER dataset that provides labeled entity information across various languages.

## 🔧 Model Training Approaches

Two different fine-tuning approaches were explored:

## 1️⃣ Full Fine-Tuning

Trained all model parameters.

- Training Loss: 0.6506 (1 epoch)
- Eval Loss: 0.4341
- F1 Score: 0.5651

## 2️⃣ LoRA Fine-Tuning

Applied Low-Rank Adaptation (LoRA) to optimize efficiency.

- Training Loss: 0.4735 (3 epochs)
- Eval Loss: 0.3514
- F1 Score: 0.6561 ✅ (Better than full fine-tuning)

## 📊 Key Results

Full Fine-tuning
- Train loss: 0.6506 (1 epoch)
- Eval loss: 0.4341
- F1 score: 0.5651

LoRA Fine-tuning
- Train loss: 0.4735 (3 epochs)
- Eval loss: 0.3514
- F1 score: 0.6561 ✅

## Why LoRA?

✔ More efficient with fewer trainable parameters.✔ Outperforms full fine-tuning in F1 Score (+9%).✔ Lower computational cost and memory usage.

📌 Training Configuration
```
training_args_lora = TrainingArguments(
    output_dir="./xlm-roberta-ner-lora",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    optim="adamw_torch",
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine"
)
```
## 🏆 Conclusion

LoRA fine-tuning proved to be superior for Multilingual NER, achieving a higher F1 score with lower computational cost.
