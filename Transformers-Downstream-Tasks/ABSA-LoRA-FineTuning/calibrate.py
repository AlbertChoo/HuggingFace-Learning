import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from temperature_scaling import ModelWithTemperature
from data_preprocessing import tokenize_dataset, load_data

# Load and tokenize dataset
dataset = load_data("Laptop_Train_v2.xml", "Restaurants_Train_v2.xml")
tokenized_dataset = tokenize_dataset(dataset)

# Prepare validation DataLoader
validation_dataloader = DataLoader(
    tokenized_dataset["validation"], batch_size=32, shuffle=False
)

# Load trained model
model = torch.load("./results/model")
model.eval()

# Apply temperature scaling
calibrated_model = ModelWithTemperature(model)
calibrated_model.to("cuda")
calibrated_model.set_temperature(validation_dataloader)

# Save the calibrated model
torch.save(calibrated_model, "./results/calibrated_model")
print("Temperature scaling completed and model saved.")
