import numpy as np
import torch
import torch.nn as nn

class VariationalAutoEncoder(nn.Module):
    def __init__(self, batch_size, latent_size):
        super(VariationalAutoEncoder, self).__init__()
        self.batch_size = batch_size
        self.latent_size = latent_size

        # Encoding Block #
        self.enc_fc1 = nn.Linear(2048, 1024)
        self.enc_bn1 = nn.BatchNorm1d(1024)
        self.enc_fc2 = nn.Linear(1024, 512)
        self.enc_bn2 = nn.BatchNorm1d(512)
        self.enc_fc3 = nn.Linear(512, 128)
        self.enc_bn3 = nn.BatchNorm1d(128)

        self.enc_mu = nn.Linear(128, self.latent_size)
        self.enc_logvar = nn.Linear(128, self.latent_size)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Decoding #
        self.dec_fc1 = nn.Linear(self.latent_size, 128)
        self.dec_bn1 = nn.BatchNorm1d(128)
        self.dec_fc2 = nn.Linear(128, 512)
        self.dec_bn2 = nn.BatchNorm1d(512)
        self.dec_fc3 = nn.Linear(512, 2048)
        self.dec_bn3 = nn.BatchNorm1d(2048)

        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

        self.dropout = nn.Dropout(p=0.2)

        # Initialize weights via Xavier Method
        self.initialize_weights()

        
    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.GRU):
                for name, param in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
                        
    def encode(self, prev_x):
        # Prev_X is of shape [batch_size, 16, 128]
        data = torch.flatten(prev_x, start_dim=1) # [batch_size, 2048]

        # Go through FC with 512 neurons, followed by RELU
        h0 = self.enc_fc1(data) # from [b, 2048] to [b, 1024]
        h0 = self.enc_bn1(h0)
        h0 = self.tanh(h0)

        h1 = self.enc_fc2(h0) # from [b, 1024] to [b, 512]
        h1 = self.enc_bn2(h1)
        h1 = self.tanh(h1)
        h1 = self.dropout(h1)

        h2 = self.enc_fc3(h1)
        h2 = self.enc_bn3(h2)
        h2 = self.tanh(h2)


        # Parallel mu and logvar generation, followed by RELU.
        mu = self.enc_mu(h2) # from [b, 128] to [b, 16]
        mu = self.relu(mu)

        log_var = self.enc_logvar(h2) # from [b, 128] to [b, 16]
        log_var = self.relu(log_var)
        log_var = log_var + 1

        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # Standard deviation
        eps = torch.randn_like(std)     # Sample epsilon from standard normal distribution
        z = mu + eps * std              # Reparameterization trick
        return z

    def decode(self, z):
        h0 = self.dec_fc1(z)
        h0 = self.dec_bn1(h0)
        h0 = self.relu(h0)
        h0 = self.dropout(h0)

        h1 = self.dec_fc2(h0)
        h1 = self.dec_bn2(h1)
        h1 = self.relu(h1)

        h2 = self.dec_fc3(h1)
        h2 = h2.view(self.batch_size, 16, 128)
        output = self.sigmoid(h2)
        return output

    def forward(self, prev_x):
        mu, log_var = self.encode(prev_x)

        z = self.reparameterize(mu, log_var)

        generated_bar = self.decode(z)
        return generated_bar, mu, log_var