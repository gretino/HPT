import os
import glob
import json
import random
import subprocess
import shutil
from pprint import pprint
from ecg_datasets import ECGDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
import ast
import numpy as np
import wfdb
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import hydra
from omegaconf import OmegaConf
from hpt.models.policy import Policy
import torch.nn as nn
import torch.optim as optim
import pickle


lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
sampling_rate=100
start_time = 0
time = 10
start_length = int(start_time * sampling_rate)
sample_length = int(time * sampling_rate)
end_time = start_time + time
t = np.arange(start_time, end_time, 1 / sampling_rate)
data_path = 'repo/HPT/'

def print_ts(obj):
    print(type(obj))
    print(obj.shape)
    
ecgdata_filtered = None
labels_filtered_encoded = None
print(os.getcwd())
# Check and load ecgdata_filtered
if os.path.exists(data_path+'ecgdata_filtered.pk'):
    with open(data_path+'ecgdata_filtered.pk', 'rb') as f:
        ecgdata_filtered = pickle.load(f)
    print("Loaded ecgdata_filtered from file.")
else:
    print("File 'ecgdata_filtered.pk' does not exist.")

# Check and load labels_filtered_encoded
if os.path.exists(data_path+'labels_filtered_encoded.pk'):
    with open(data_path+'labels_filtered_encoded.pk', 'rb') as f:
        labels_filtered_encoded = pickle.load(f)
    print("Loaded labels_filtered_encoded from file.")
else:
    print("File 'labels_filtered_encoded.pk' does not exist.")
    
ecg_dataset = ECGDataset(ecgdata_filtered, labels_filtered_encoded)
train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(ecg_dataset, [0.7, 0.2, 0.1])

normalizer = ecg_dataset.get_normalizer()


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
cfg = OmegaConf.load(data_path+'experiments/configs/config.yaml')
policy = Policy(embed_dim=128, num_blocks=4, num_heads=4, action_horizon=0, token_postprocessing="max")
# Assuming 'ecg_dataset' is your domain name
policy.init_domain_stem(domain_name='ecg_dataset', stem_spec=cfg.stem_ecg)
# Initialize domain head with the normalizer
policy.init_domain_head(
    domain_name='ecg_dataset',
    normalizer=normalizer,
    head_spec=cfg.head_ecg
)
policy.finalize_modules()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
policy.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(policy.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

num_epochs = cfg.train.total_epochs

for epoch in range(num_epochs):
    policy.train()
    running_loss = 0.0
    for batch in train_loader:
        #inputs = batch['data'].to(device).float()  # Shape: (batch_size, sequence_length, input_dim)
        #labels = batch['label'].to(device).long()  # Shape: (batch_size,)
        
        #print_ts(inputs)
        #print_ts(labels)
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = policy.compute_loss(batch)  # Adjust based on your model's expected input
        # Backward pass and optimization
        outputs.backward()
        optimizer.step()
        
        running_loss += outputs.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Optional: Validate the model
    # policy.eval()
    # with torch.no_grad():
    #     # Perform validation
