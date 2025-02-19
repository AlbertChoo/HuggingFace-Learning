# Text-to-SQL using T5 with QLoRA & Full Fine-Tuning

This project fine-tunes a `T5-small` model for text-to-SQL generation using the Spider dataset. It explores both **QLoRA-based optimization** and **full fine-tuning** to improve model performance.

## Key Features
- Implements `T5-small` model for text-to-SQL conversion.
- Prepares datasets with explicit instruction-based prompts.
- Fine-tunes using **QLoRA** for memory-efficient training and **full fine-tuning** for best performance.
- Evaluates performance using **ROUGE** metrics.

## Implementation Overview
The entire pipeline, from data preprocessing to training and evaluation, is implemented in a single Jupyter Notebook: **`Text_to_SQL_T5small_QLoRA_FullFineTuning.ipynb`**.

## Setup

### 1. Install Dependencies
Ensure the required dependencies are installed:
```bash
pip install -U datasets evaluate transformers torch peft
```

### 2. Prepare the Dataset
- The dataset is preprocessed into instruction-based prompts suitable for large language models.
- Tokenization is applied to convert prompt-response pairs into `input_ids` for model training.

### 3. Train the Model
- **QLoRA Fine-Tuning:** Implemented but not fully trained due to long training time on Google Colab’s T4 GPU (~7 hours per epoch).
- **Full Fine-Tuning:** Successfully trained with the dataset on a laptop with an **RTX 3070 GPU**.

#### Full Fine-Tuning Results:
- **Epochs:** 2
- **Learning Rate:** 5e-3
- **Training Time:** 2h 49m 1s
- **Training Loss:** 0.023100
- **Validation Loss:** 0.013285

### 4. Evaluate the Model
Evaluation uses **ROUGE metrics**:

#### Original (Base) Model:
- ROUGE-1: 0.0319
- ROUGE-2: 0.0050
- ROUGE-L: 0.0307
- ROUGE-Lsum: 0.0312

#### Fine-Tuned Model:
- ROUGE-1: 0.9233
- ROUGE-2: 0.8863
- ROUGE-L: 0.9176
- ROUGE-Lsum: 0.9182

## Notes
- The full fine-tuning results and evaluation metrics are referenced from an article on **[Medium](https://medium.com/@martinkeywood/fine-tuning-a-t5-small-model-to-generate-sql-from-natural-language-with-92-3-accuracy-fb29e062c638)**.
- The original author's GitHub repository: **[GitHub](https://gist.github.com/mkeywood1/9e8411aef44cf18009aa3e4776501c08)**.
- Hyperparameters and dataset configurations can be adjusted in the notebook for further tuning.

---
This project is implemented within a single Jupyter Notebook, making it easy to follow and experiment with different fine-tuning approaches.
