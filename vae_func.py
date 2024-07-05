import torch
import torch.nn as nn

def vae_gaussian_kl_loss(mu, log_var):
  KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
  return KLD.mean()

def reconstruction_loss(x_reconstructed, x):
  bce_loss = nn.BCELoss()
  return bce_loss(x_reconstructed, x)

def vae_loss(y_pred, y_true):
  mu, log_var, reconstructed_x = y_pred
  reconstructed_loss = reconstruction_loss(reconstructed_x, y_true)
  kld_loss = vae_gaussian_kl_loss(mu, log_var)

  return reconstructed_loss + kld_loss