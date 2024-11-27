
from torch import nn
import torch
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, p, p_cond, latent_dim, p_encoder_lst, hidden_act='elu', use_batch_norm=True):
        super(Encoder, self).__init__()
        self.p = p
        self.p_cond = p_cond
        self.latent_dim = latent_dim
        self.use_batch_norm = use_batch_norm

        layers = []
        input_dim = p + p_cond
        if use_batch_norm:
            for hidden_dim in p_encoder_lst:
                layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(self.get_activation(hidden_act))
                input_dim = hidden_dim
            
            self.z_mean = nn.Sequential(
                nn.Linear(input_dim, latent_dim, bias=False),
                nn.BatchNorm1d(latent_dim)
            )
            self.z_log_var = nn.Sequential(
                nn.Linear(input_dim, latent_dim, bias=False),
                nn.BatchNorm1d(latent_dim)
            )
        else:
            for hidden_dim in p_encoder_lst:
                layers.append(nn.Linear(input_dim, hidden_dim, bias=True))
                layers.append(self.get_activation(hidden_act))
                input_dim = hidden_dim
            
            self.z_mean = nn.Linear(input_dim, latent_dim, bias=True)
            self.z_log_var = nn.Linear(input_dim, latent_dim, bias=True)
              

        self.encoder = nn.Sequential(*layers)


    def get_activation(self, act_name):
        if act_name == 'elu':
            return nn.ELU()
        elif act_name == 'relu':
            return nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {act_name}")

    def forward(self, x, cond):
        x = torch.cat((x, cond), dim=1)
        h = self.encoder(x)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        return z_mean, z_log_var


class Decoder(nn.Module):
    def __init__(self, p, p_cond, latent_dim, p_decoder_lst, hidden_act='elu', output_act='relu', use_batch_norm=True):
        super(Decoder, self).__init__()
        self.p = p
        self.p_cond = p_cond
        self.latent_dim = latent_dim
        self.use_batch_norm = use_batch_norm

        layers = []
        input_dim = latent_dim + p_cond
        if use_batch_norm:
            for hidden_dim in p_decoder_lst:
                layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(self.get_activation(hidden_act))
                input_dim = hidden_dim

            layers.append(nn.Linear(input_dim, p, bias=False))
            layers.append(nn.BatchNorm1d(p))
            layers.append(self.get_activation(output_act))
        else:
            for hidden_dim in p_decoder_lst:
                layers.append(nn.Linear(input_dim, hidden_dim, bias=True))
                layers.append(self.get_activation(hidden_act))
                input_dim = hidden_dim

            layers.append(nn.Linear(input_dim, p, bias=True))
            layers.append(self.get_activation(output_act))

        self.decoder = nn.Sequential(*layers)

    def get_activation(self, act_name):
        if act_name == 'elu':
            return nn.ELU()
        elif act_name == 'relu':
            return nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {act_name}")

    def forward(self, z, cond):
        z = torch.cat((z, cond), dim=1)
        return self.decoder(z)


class CVAE(nn.Module):
    def __init__(self, p, p_cond, latent_dim, p_encoder_lst, p_decoder_lst, hidden_act='elu', output_act='relu', use_batch_norm=True):
        super(CVAE, self).__init__()
        self.encoder = Encoder(p, p_cond, latent_dim, p_encoder_lst, hidden_act, use_batch_norm)
        self.decoder = Decoder(p, p_cond, latent_dim, p_decoder_lst, hidden_act, output_act, use_batch_norm)

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x, cond):
        z_mean, z_log_var = self.encoder(x, cond)
        z = self.reparameterize(z_mean, z_log_var)
        x_reconstructed = self.decoder(z, cond)
        return x_reconstructed, z_mean, z_log_var



def loss_function(x, x_reconstructed, z_mean, z_log_var):

    reconstruction_loss = F.mse_loss(x_reconstructed, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return reconstruction_loss + kl_loss, reconstruction_loss, kl_loss




def weighted_loss_function(x, x_reconstructed, z_mean, z_log_var, weights):

    reconstruction_loss = F.mse_loss(x_reconstructed, x, reduction='none')
    reconstruction_loss = reconstruction_loss.sum(dim=1) 
    reconstruction_loss = reconstruction_loss * weights 
    reconstruction_loss = reconstruction_loss.sum()  
    
    kl_loss = -0.5 * (1 + z_log_var - z_mean.pow(2) - z_log_var.exp()).sum(dim=1)
    kl_loss = kl_loss * weights 
    kl_loss = kl_loss.sum()  

    total_loss = reconstruction_loss + kl_loss

    return total_loss, reconstruction_loss, kl_loss