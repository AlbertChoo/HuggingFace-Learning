Main Components:
## Introduction

1. Covers fine-tuning techniques:
- Language modeling
- Supervised Fine-Tuning (SFT)
- Preference Fine-Tuning
- Uses Parameter Efficient Fine-Tuning (PEFT) for optimization.

2. Dependencies & Dataset
- Installs required packages: accelerate, bitsandbytes, trl, peft, transformers, datasets
- Loads and processes the dataset.

3. Model Loading
- Loads the Phi-2/Phi-3 model using Hugging Face Transformers.
- Utilizes bitsandbytes for 4-bit quantization (reducing memory usage).

4. Fine-Tuning Setup
- Implements PEFT to fine-tune the model with fewer trainable parameters.
- Uses LoRA (Low-Rank Adaptation) for efficient training.

5. Training & Evaluation
- Defines training arguments.
- Runs fine-tuning with Hugging Face's Trainer API.
- Saves and loads the fine-tuned model.
