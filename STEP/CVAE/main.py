import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from .data import load_data
from .model import *


def set_random_seed(seed):
    """Set all relevant random seeds to ensure reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Basic early-stopping utility to halt training when validation loss plateaus."""

    def __init__(self, min_delta=0, patience=30, restore_best_weights=True, verbose=False):
        self.min_delta = min_delta
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        current_score = -val_loss  # Higher score means lower loss

        if self.best_score is None:
            # First evaluation: store as the current best
            self.best_score = current_score
            self.best_model_state = model.state_dict()
        elif current_score < self.best_score + self.min_delta:
            # Not enough improvement: increase patience counter
            self.epochs_no_improve += 1
            if self.verbose:
                print(f'EarlyStopping: No improvement in {self.epochs_no_improve}/{self.patience} epochs')
            if self.epochs_no_improve >= self.patience:
                if self.verbose:
                    print('EarlyStopping: Stopping training')
                self.early_stop = True
                if self.restore_best_weights:
                    if self.verbose:
                        print('EarlyStopping: Restoring best model weights')
                    model.load_state_dict(self.best_model_state)
        else:
            # Improvement observed: reset counter and save model state
            self.best_score = current_score
            self.best_model_state = model.state_dict()
            self.epochs_no_improve = 0


class Args:
    """Default argument container for model training; override as needed."""

    def __init__(self):
        self.lr = 0.01
        self.bs = 16384
        self.k_sc_max = 8         # Max number of cells per pseudo-spot
        self.k_sc_min = 2         # Min number of cells per pseudo-spot
        self.n_samples = 50000    # Total pseudo-spot budget
        self.device = 'cuda:2'
        self.k_st_min = 2
        self.k_st_max = 6
        self.input_max = 10
        self.num_hidden_layer = 2
        self.use_batch_norm = True
        self.epoch = 200
        self.output_act = 'relu'
        self.hidden_act = 'elu'
        self.beta = 1           # KL weight (beta-VAE); < 1 relaxes regularization


def Reconstruct(encoder, decoder, mat_sc_r, mat_sp_r, minmax, args, image_based=False):
    """
    Use trained encoder/decoder to generate condition-specific reconstructions
    for both ST and SC inputs.
    """
    set_random_seed(30)

    def recon(x, cond, cond_=None):
        """Encode input, sample latent variable, and decode under new condition."""
        if cond_ is None:
            cond_ = cond.clone()
        z_mean, z_log_var = encoder(x, cond)
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        z = z_mean + eps * std
        x_reconstructed = decoder(z, cond_)
        return x_reconstructed

    # ----- Spatial reconstruction (change condition from input_max to 0) -----
    x = torch.tensor(mat_sp_r, dtype=torch.float32).to(args.device)
    cond = torch.full((mat_sp_r.shape[0], 1), args.input_max, dtype=torch.float32, device=args.device)
    cond_ = torch.full((mat_sp_r.shape[0], 1), 0, dtype=torch.float32, device=args.device)
    new_st = recon(x, cond, cond_)
    new_st = new_st.detach().cpu().numpy()
    new_st = minmax.inverse_transform(new_st)
    if not image_based:
        new_st = np.expm1(new_st)
    new_st = np.nan_to_num(new_st, nan=0.0, posinf=0.0, neginf=0.0)

    # ----- scRNA reconstruction (keep condition at 0) -----
    x = torch.tensor(mat_sc_r, dtype=torch.float32).to(args.device)
    cond = torch.full((mat_sc_r.shape[0], 1), 0, dtype=torch.float32, device=args.device)
    new_sc = recon(x, cond)
    new_sc = new_sc.detach().cpu().numpy()
    new_sc = minmax.inverse_transform(new_sc)
    if not image_based:
        new_sc = np.expm1(new_sc)
    new_sc = np.nan_to_num(new_sc, nan=0.0, posinf=0.0, neginf=0.0)

    return new_st, new_sc


def DomainAdaptation(adata_st, adata_sc, n_celltype, outdir, args=None, loggings=None, image_based=False):
    """
    Main pipeline:
    1. Prepare ST/SC data and pseudo-samples.
    2. Train the conditional VAE.
    3. Plot training loss.
    4. Reconstruct modality-specific outputs.
    """
    if args is None:
        args = Args()

    assert adata_sc.shape[1] == adata_st.shape[1], "ST and SC must share the same gene dimension"
    if image_based:
        loggings.info("The input is Image-based dataset! We will not perform log1p operations!")

    # Build datasets and loader; obtain scaled matrices plus SC scaler
    p, loader, mat_sc_r, mat_sp_r, minmax_sc = load_data(
        adata_st, adata_sc, n_celltype, args, image_based, loggings
    )

    p_cond = 1
    latent_dim = n_celltype * 3
    # Construct hidden sizes via geometric progression from latent_dim to input dim
    hidden_dim = list(
        np.floor(
            np.geomspace(latent_dim, p, args.num_hidden_layer + 2)[1:args.num_hidden_layer + 1]
        ).astype('int')
    )
    cvae = CVAE(
        p, p_cond, latent_dim, hidden_dim[::-1], hidden_dim,
        hidden_act=args.hidden_act, output_act=args.output_act, use_batch_norm=args.use_batch_norm
    ).to(args.device)
    optimizer = optim.Adam(cvae.parameters(), lr=args.lr)

    trainLoss = []
    for epoch in tqdm(range(args.epoch)):
        cvae.train()
        all_loss = 0.0
        for batch_data, batch_labels, batch_weights in loader:
            if batch_data.shape[0] <= 1:
                continue
            batch_data = batch_data.to(args.device)
            batch_labels = batch_labels.to(args.device)
            batch_weights = batch_weights.to(args.device)

            optimizer.zero_grad()
            x_reconstructed, z_mean, z_log_var = cvae(batch_data, batch_labels)
            loss = weighted_loss_function(batch_data, x_reconstructed, z_mean, z_log_var, batch_weights, beta=args.beta)[0]
            all_loss += loss.item()
            loss.backward()
            optimizer.step()

        trainLoss.append(all_loss / len(loader))

    # Plot and save the training loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(trainLoss, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(outdir + '/model/train_loss.png')
    plt.close()

    cvae.eval()

    # Instantiate standalone encoder/decoder for later reuse and copy trained weights
    encoder = Encoder(
        p, p_cond, latent_dim, hidden_dim[::-1],
        hidden_act=args.hidden_act, use_batch_norm=args.use_batch_norm
    ).to(args.device)
    decoder = Decoder(
        p, p_cond, latent_dim, hidden_dim,
        hidden_act=args.hidden_act, output_act=args.output_act, use_batch_norm=args.use_batch_norm
    ).to(args.device)

    encoder.load_state_dict(cvae.encoder.state_dict())
    decoder.load_state_dict(cvae.decoder.state_dict())

    encoder.eval()
    decoder.eval()

    # Generate reconstructed ST/SC matrices
    new_st, new_sc = Reconstruct(encoder, decoder, mat_sc_r, mat_sp_r, minmax_sc, args, image_based)

    return cvae, new_st, new_sc