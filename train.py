import numpy as np
import torch
import torch.nn as nn

from model import VariationalAutoEncoder
from dataset import HooktheoryDataset
from vae_func import vae_loss

##################
# HYPERPARAMETERS #
EPOCHS = 20
BATCH_SIZE = 10
LATENT_SIZE = 8
LEARNING_RATE = 0.001
###################

##################
# REQUIRED FUNCTIONS #
def train_one_epoch(epoch_index, dataloader, optimizer):
  for i, data in enumerate(dataloader):
    prev_x, current_x = data

    optimizer.zero_grad()

    generated_bar, mu, log_var = model(prev_x)

    loss = vae_loss((mu, log_var, generated_bar), current_x, epoch_index)
    print(f"Epoch {epoch_index} Loss {i}: {loss.item()}")
    loss.backward()
    optimizer.step()
###################


##################
# Initialization # 
dataset = HooktheoryDataset()
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VariationalAutoEncoder(BATCH_SIZE, LATENT_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
###################

##################
# TRAINING LOOP #
for epoch in range(EPOCHS):
  model.train(True)
  train_one_epoch(epoch, train_dataloader, optimizer)
##################