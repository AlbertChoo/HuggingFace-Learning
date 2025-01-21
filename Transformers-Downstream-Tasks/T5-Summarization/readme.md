Dataset:
from datasets import load_dataset
dataset = load_dataset('cnn_dailymail', '3.0.0', split='train[:10]')
# We take a part of train dataset from 'cnn_dailymail'

Model Used: 't5-small', AutoTokenizer, AutoModelForSeq2SeqLM
