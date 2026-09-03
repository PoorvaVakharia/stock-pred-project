import torch
from src.autoencoder import Autoencoder
import config

def build_model(input_dim, latent_dim=8, hidden_dims=None):
    if hidden_dims is None:
        hidden_dims = [max(16, 2*latent_dim)]
    return Autoencoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(input_dim, latent_dim, hidden_dims, path):
    model = Autoencoder(input_dim, latent_dim, hidden_dims)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model
