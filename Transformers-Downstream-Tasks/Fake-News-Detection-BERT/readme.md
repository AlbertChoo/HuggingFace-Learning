Dataset:
"https://github.com/laxmimerit/All-CSV-ML-Data-Files-Download/raw/master/fake_news.xlsx"

Model Used: "distilbert-base-uncased", "google/mobilebert-uncased", "huawei-noah/TinyBERT_General_4L_312D", "bert-base-uncased"

Training Code:
'''
BATCH_SIZE=32
training_dir = 'train_dir'

training_args = TrainingArguments(
    output_dir = training_dir,
    overwrite_output_dir = True,
    num_train_epochs = 2,
    learning_rate = 2e-5,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size = BATCH_SIZE,
    weight_decay = 0.01,
    evaluation_strategy = 'epoch'
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = encoded_dataset['train'],
    eval_dataset = encoded_dataset['validation'],
    compute_metrics = compute_metrics,
    tokenizer = distilbert_tokenizer
)

trainer.train()
'''
Model Performance:
'
{'bert-base': {'bert-base': {'test_loss': 0.19906283915042877,
   'test_accuracy': 0.962253829321663,
   'test_f1': 0.9622883013603741,
   'test_runtime': 19.5463,
   'test_samples_per_second': 187.043,
   'test_steps_per_second': 23.38},
  'time taken': 1029.7146208286285},
 'distilbert': {'distilbert': {'test_loss': 0.1682099848985672,
   'test_accuracy': 0.962800875273523,
   'test_f1': 0.9628468639291917,
   'test_runtime': 10.129,
   'test_samples_per_second': 360.945,
   'test_steps_per_second': 45.118},
  'time taken': 677.7424376010895},
 'mobilebert': {'mobilebert': {'test_loss': 0.17855119705200195,
   'test_accuracy': 0.962800875273523,
   'test_f1': 0.9628170461473012,
   'test_runtime': 18.4213,
   'test_samples_per_second': 198.466,
   'test_steps_per_second': 24.808},
  'time taken': 890.103978395462},
 'tinybert': {'tinybert': {'test_loss': 0.19918876886367798,
   'test_accuracy': 0.9584245076586433,
   'test_f1': 0.9584182733647391,
   'test_runtime': 3.4344,
   'test_samples_per_second': 1064.513,
   'test_steps_per_second': 133.064},
  'time taken': 116.63017511367798}}
'
