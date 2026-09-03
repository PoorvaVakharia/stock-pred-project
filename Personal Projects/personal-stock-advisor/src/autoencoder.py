import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [max(16, 2*latent_dim)]
        # Encoder
        enc = []
        last = input_dim
        for h in hidden_dims:
            enc.append(nn.Linear(last, h))
            enc.append(nn.ReLU())
            last = h
        enc.append(nn.Linear(last, latent_dim))
        self.encoder = nn.Sequential(*enc)
        # Decoder
        dec = []
        last = latent_dim
        for h in reversed(hidden_dims):
            dec.append(nn.Linear(last, h))
            dec.append(nn.ReLU())
            last = h
        dec.append(nn.Linear(last, input_dim))
        self.decoder = nn.Sequential(*dec)
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

def save_autoencoder(model, path):
    torch.save(model.state_dict(), path)

def load_autoencoder(path, input_dim=None, latent_dim=8, hidden_dims=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Try to infer input_dim from saved config if not provided
    import config, json, os
    if input_dim is None:
        if os.path.exists(config.CONFIG_SAVE_PATH):
            with open(config.CONFIG_SAVE_PATH) as f:
                conf = json.load(f)
            input_dim = conf["input_dim"]
            latent_dim = conf["latent_dim"]
            hidden_dims = conf["hidden_dims"]
        else:
            raise ValueError("input_dim must be specified or config file present.")
    model = Autoencoder(input_dim, latent_dim, hidden_dims)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model, device
