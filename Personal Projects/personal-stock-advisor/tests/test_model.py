import torch
from src.autoencoder import Autoencoder

def test_autoencoder_forward():
    model = Autoencoder(input_dim=5, latent_dim=2, hidden_dims=[8])
    x = torch.randn(3, 5)
    recon, latent = model(x)
    assert recon.shape == (3, 5)
    assert latent.shape == (3, 2)
    loss = ((x - recon) ** 2).mean()
    assert loss.item() >= 0
