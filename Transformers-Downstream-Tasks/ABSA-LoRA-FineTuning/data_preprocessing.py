import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset
from transformers import BertTokenizer

# Function to parse ABSA XML dataset
def parse_absa_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []
    for sentence in root.findall("sentence"):
        text = sentence.find("text").text
        aspects = sentence.find("aspectTerms")
        if aspects is not None:
            for aspect in aspects.findall("aspectTerm"):
                aspect_term = aspect.get("term")
                polarity = aspect.get("polarity")
                data.append([text, aspect_term, polarity])
    return pd.DataFrame(data, columns=["sentence", "aspect", "sentiment"])

# Load and preprocess dataset
def load_data(laptop_path, restaurant_path):
    laptop_df = parse_absa_xml(laptop_path)
    restaurant_df = parse_absa_xml(restaurant_path)
    absa_df = pd.concat([laptop_df, restaurant_df], ignore_index=True)
    
    # Data Cleaning
    absa_df['sentence'] = absa_df['sentence'].str.lower().str.strip()
    sentiment_mapping = {'positive': 1, 'negative': 0, 'neutral': 2}
    absa_df['sentiment_encoded'] = absa_df['sentiment'].map(sentiment_mapping)
    
    # Split dataset
    train_df, test_df = train_test_split(absa_df, test_size=0.2, stratify=absa_df['sentiment'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['sentiment'], random_state=42)
    
    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True))
    })
    return dataset_dict

# Tokenization
def tokenize_dataset(dataset_dict):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    def tokenize_function(example):
        return tokenizer(example["sentence"], padding="max_length", truncation=True, max_length=128)
    return dataset_dict.map(tokenize_function, batched=True)

if __name__ == "__main__":
    dataset = load_data("Laptop_Train_v2.xml", "Restaurants_Train_v2.xml")
    tokenized_dataset = tokenize_dataset(dataset)
    print("Dataset prepared and tokenized.")
