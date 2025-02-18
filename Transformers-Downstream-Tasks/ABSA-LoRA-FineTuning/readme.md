# Aspect-Based Sentiment Analysis (ABSA) with LoRA Fine-Tuning

## Overview
This project implements Aspect-Based Sentiment Analysis (ABSA) using BERT with LoRA fine-tuning on the SemEval-2014 Task 4 dataset. The model classifies sentiments (Positive, Negative, Neutral) for specific aspects within text. To improve probability calibration, temperature scaling was applied post-training.

## Features
- **LoRA Fine-Tuning**: Efficient adaptation of BERT with trainable low-rank adapters.
- **Aspect-Based Sentiment Classification**: Sentiment analysis at the aspect level rather than full sentence.
- **Hugging Face Integration**: Tokenization and dataset preparation with datasets and transformers.
- **Temperature Scaling**: Post-hoc calibration for better probability reliability.

## Dataset
- **Source**: [[SemEval-2014 Task 4]](https://paperswithcode.com/dataset/semeval-2014-task-4-sub-task-2)
- **Categories**: Laptops & Restaurants
- **Labels**: Positive, Negative, Neutral

## Requirements
```
torch==2.1.0
transformers==4.36.0
datasets==2.14.5
peft==0.7.1
scikit-learn==1.3.0
numpy==1.23.5
pandas==1.5.3
tqdm==4.66.1
accelerate==0.24.0
```

## Installation
```pip install -U transformers accelerate datasets torch```

## Data Preparation
1. **Parse XML Files**: Extract aspect terms and their sentiment labels.
2. **Clean & Normalize**: Remove special characters, lowercase text, and trim whitespace.
3. **Tokenize**: Apply BERT tokenizer (bert-base-uncased) with padding and truncation.
4. **Dataset Split**: Train (80%), Validation (10%), Test (10%).

## Model Training
- **Base Model**: bert-base-uncased
- **Fine-Tuning**: Applied LoRA on query, key, value, and dense layers.
- **Hyperparameters**:
  - Learning Rate: 2e-4
  - Batch Size: 2
  - Epochs: 5
  - Optimizer: AdamW
  - Weight Decay: 0.001

## Evaluation Metrics
| Metric | Before Calibration  | After Calibration |
| :---:   | :---: | :---: |
| Accuracy | 0.74   | 0.74   |
| Precision | 0.74   | 0.74   |
| Recall | 0.75   | 0.75   |
| F1-Score | 0.75   | 0.75   |
| ECE (Expected Calibration Error) | 0.112   | 0.000   |

## Temperature Scaling
- **Goal**: Adjust model confidence scores to better match real accuracy.
- **Technique**: Optimized a scalar T using Negative Log Likelihood (NLL) loss.
- **Result**: Reduced ECE from 0.112 to 0.000, improving reliability of probability outputs.

## Usage
**Training the Model**

```python train.py --data <path_to_data> --save <save_folder>```

**Applying Temperature Scaling**

```python calibrate.py --data <path_to_data> --save <save_folder>```

## Results & Insights
- **LoRA fine-tuning performed similarly to full fine-tuning** while being more memory efficient.
- **Probability calibration significantly improved** after temperature scaling, while F1-score remained the same.

- **Potential Next Steps**:
  - Experiment with different architectures (RoBERTa, T5).
  - Increase dataset size for better generalization.
  - Implement other calibration techniques (Platt scaling, isotonic regression).

## Author
[Vernon Choo Chee Yang]
- [[LinkedIn Profile]](https://www.linkedin.com/in/vernon-choo-chee-yang-61a966247/)
- [[GitHub Profile]](https://github.com/AlbertChoo?tab=repositories) ```Inside HuggingFace Learning```

## Acknowledgments
- Hugging Face Transformers & Datasets
- SemEval 2014 Task 4 Dataset
- LoRA Implementation by PEFT Library
