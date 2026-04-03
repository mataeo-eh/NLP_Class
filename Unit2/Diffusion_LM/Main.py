"""
Diffusion-Based Language Model — trained on Shakespeare plays.

Architecture overview:
  A discrete diffusion language model that works by:
  1. Embedding token sequences into continuous space
  2. Adding Gaussian noise over T diffusion timesteps (forward process)
  3. Training a small Transformer to predict the clean embeddings from noisy ones
  4. At generation time, starting from pure noise and iteratively denoising

This is a simplified/educational implementation — not production-grade, but enough
to see how diffusion applies to discrete text and to generate Shakespeare-flavored output.

Key references:
  - "Diffusion-LM Improves Controllable Text Generation" (Li et al., 2022)
  - "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
"""

import math
import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Device setup — Apple Silicon MPS when available, else CPU
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


# ===========================================================================
# 1. DATA — Load Shakespeare corpus and build a character-level tokenizer
# ===========================================================================
# We use character-level tokenization for simplicity. A diffusion LM works on
# continuous embeddings, so the vocab size matters less than in autoregressive
# models — we just need a way to map text <-> discrete tokens.

# Path to the Shakespeare CSV relative to the project root
SHAKESPEARE_CSV = os.path.join(
    os.path.dirname(__file__), "..", "..", "Unit1", "Week8", "shakespeare_plays.csv"
)


def load_shakespeare_text(csv_path: str) -> str:
    """
    Read the Shakespeare CSV and concatenate all dialogue text into one string.
    The CSV has columns: index, play_name, genre, character, act, scene, sentence, text, sex
    We only care about the 'text' column (index 7, 0-based).
    """
    lines = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for row in reader:
            if len(row) > 7:
                lines.append(row[7].strip())
    # Join all lines with newlines to preserve some structure
    return "\n".join(lines)


class CharTokenizer:
    """
    Minimal character-level tokenizer.
    Maps each unique character to an integer ID. Reserves ID 0 for <PAD>.
    """

    def __init__(self, text: str):
        # Build vocabulary from all unique characters in the corpus
        chars = sorted(set(text))
        # Reserve index 0 for padding
        self.char_to_id = {ch: i + 1 for i, ch in enumerate(chars)}
        self.id_to_char = {i + 1: ch for i, ch in enumerate(chars)}
        self.pad_id = 0
        self.vocab_size = len(chars) + 1  # +1 for the pad token

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of token IDs."""
        return [self.char_to_id.get(ch, self.pad_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Convert a list of token IDs back to a string."""
        return "".join(self.id_to_char.get(i, "") for i in ids)


class ShakespeareDataset(Dataset):
    """
    Produces fixed-length chunks of tokenized Shakespeare text.
    Each item is a 1-D tensor of token IDs with length `seq_len`.
    """

    def __init__(self, token_ids: list[int], seq_len: int = 128):
        self.seq_len = seq_len
        # Trim corpus so it divides evenly into chunks
        n_chunks = len(token_ids) // seq_len
        self.data = torch.tensor(token_ids[: n_chunks * seq_len], dtype=torch.long)
        self.data = self.data.view(n_chunks, seq_len)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]


# ===========================================================================
# 2. DIFFUSION SCHEDULE — defines how noise is added over T timesteps
# ===========================================================================
# We use a cosine noise schedule (Nichol & Dhariwal, 2021), which provides
# more gradual noising than the original linear schedule.

class DiffusionSchedule:
    """
    Precomputes the noise schedule parameters:
      - beta_t:      variance of noise added at step t
      - alpha_t:     1 - beta_t
      - alpha_bar_t: cumulative product of alpha up to step t
                     (how much of the original signal remains at step t)

    The forward process q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0,
                                                (1 - alpha_bar_t) * I)
    """

    def __init__(self, num_timesteps: int = 1000, schedule: str = "cosine"):
        self.T = num_timesteps

        if schedule == "cosine":
            # Cosine schedule: alpha_bar_t = f(t)/f(0) where f(t) = cos^2(...)
            steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
            s = 0.008  # small offset to prevent beta from being exactly 0
            f_t = torch.cos(((steps / num_timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
            alpha_bar = f_t / f_t[0]
            # Derive betas from alpha_bar, clipping for numerical stability
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            betas = betas.clamp(max=0.999)
        else:
            # Linear schedule fallback
            betas = torch.linspace(1e-4, 0.02, num_timesteps, dtype=torch.float64)

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        # Store everything as float32 for use during training
        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alpha_bar = alpha_bar.float()
        # Precompute commonly used derived quantities
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar).float()
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar).float()

    def to(self, device):
        """Move all schedule tensors to the specified device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        return self


# ===========================================================================
# 3. TRANSFORMER DENOISER — the neural network that learns to remove noise
# ===========================================================================
# This is the core model. It takes noisy embeddings + a timestep indicator
# and predicts the clean (denoised) embeddings. We use a small Transformer
# encoder architecture (no causal mask — diffusion sees the whole sequence).

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Encodes the diffusion timestep t as a fixed sinusoidal vector, following
    the positional encoding idea from "Attention Is All You Need". This tells
    the denoiser how much noise is present so it can calibrate its prediction.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: (batch,) — integer timesteps
        half_dim = self.dim // 2
        # Frequency terms: exp(-log(10000) * i / half_dim)
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        # Outer product: (batch, 1) * (1, half_dim) -> (batch, half_dim)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        # Concatenate sin and cos to get full embedding
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb  # shape: (batch, dim)


class TransformerDenoiser(nn.Module):
    """
    A small Transformer encoder that denoises token embeddings.

    Input:  noisy embeddings (batch, seq_len, embed_dim) + timestep (batch,)
    Output: predicted clean embeddings (batch, seq_len, embed_dim)

    Architecture:
      - Token embedding layer (shared between input projection and output)
      - Sinusoidal timestep embedding, projected and added to each position
      - Learnable positional embeddings for sequence position
      - N Transformer encoder layers (bidirectional self-attention)
      - Linear output head projecting back to embedding dimension
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 1024,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Token embedding: maps discrete token IDs -> continuous vectors
        # This is the embedding space where diffusion operates
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Positional embedding: learnable, tells the model about token position
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        # Timestep embedding: sinusoidal encoding of the diffusion step,
        # projected to match embed_dim so it can be added to the sequence
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Transformer encoder layers — bidirectional attention (no causal mask)
        # because diffusion models denoise the entire sequence at once
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # input/output shape: (batch, seq, dim)
            norm_first=True,   # pre-norm architecture — more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm before the output projection
        self.out_norm = nn.LayerNorm(embed_dim)

        # Output head: projects transformer output back to embed_dim
        # (predicts the clean embedding at each position)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: noisy embeddings, shape (batch, seq_len, embed_dim)
            t:   diffusion timestep for each sample, shape (batch,)

        Returns:
            Predicted clean embeddings, shape (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x_t.shape

        # Add positional information so the model knows token order
        positions = torch.arange(seq_len, device=x_t.device)
        pos_emb = self.pos_embed(positions)  # (seq_len, embed_dim)

        # Compute timestep embedding and broadcast across all positions
        # This tells every position how much noise is present
        t_emb = self.time_embed(t)  # (batch, embed_dim)
        t_emb = t_emb.unsqueeze(1)  # (batch, 1, embed_dim)

        # Combine: noisy input + positional info + timestep info
        h = x_t + pos_emb + t_emb

        # Pass through the transformer encoder stack
        h = self.transformer(h)

        # Project to predicted clean embeddings
        h = self.out_norm(h)
        h = self.out_proj(h)

        return h


# ===========================================================================
# 4. DIFFUSION LANGUAGE MODEL — ties together the schedule, model, and logic
# ===========================================================================

class DiffusionLM(nn.Module):
    """
    Wraps the denoiser + diffusion schedule into a complete diffusion LM.

    Training:
      1. Embed clean tokens into continuous space via the token embedding
      2. Sample a random timestep t for each sequence in the batch
      3. Add noise according to the forward diffusion schedule at step t
      4. Ask the denoiser to predict the clean embeddings from the noisy ones
      5. Loss = MSE between predicted and actual clean embeddings

    Generation (reverse process):
      1. Start from pure Gaussian noise in embedding space
      2. For t = T down to 1, use the denoiser to estimate clean embeddings,
         then compute x_{t-1} using the reverse diffusion formula
      3. After reaching t=0, project the denoised embeddings to the nearest
         token in vocabulary space via cosine similarity
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 1024,
        max_seq_len: int = 128,
        num_timesteps: int = 1000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.embed_dim = embed_dim

        # The denoising Transformer
        self.denoiser = TransformerDenoiser(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Noise schedule — precomputed alpha/beta values
        self.schedule = DiffusionSchedule(num_timesteps, schedule="cosine")

    def to(self, device):
        """Override to also move the schedule tensors to the device."""
        self.schedule.to(device)
        return super().to(device)

    def get_clean_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up the embedding for each token ID. These are the 'clean' x_0
        targets that the diffusion process will try to recover.
        """
        return self.denoiser.token_embed(token_ids)

    def forward_diffusion(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The forward (noising) process: q(x_t | x_0).
        Adds noise to clean embeddings according to the schedule at timestep t.

        Returns:
            x_t:   the noisy embeddings at timestep t
            noise: the actual noise that was added (used as alternative training target)
        """
        # Gather the schedule values for the sampled timesteps
        # Each sample in the batch may have a different timestep
        sqrt_ab = self.schedule.sqrt_alpha_bar[t]          # (batch,)
        sqrt_1m_ab = self.schedule.sqrt_one_minus_alpha_bar[t]  # (batch,)

        # Reshape for broadcasting: (batch,) -> (batch, 1, 1)
        sqrt_ab = sqrt_ab[:, None, None]
        sqrt_1m_ab = sqrt_1m_ab[:, None, None]

        # Sample random Gaussian noise
        noise = torch.randn_like(x_0)

        # Closed-form forward process: x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*noise
        x_t = sqrt_ab * x_0 + sqrt_1m_ab * noise

        return x_t, noise

    def compute_loss(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Training loss for one batch.

        We use the "predict x_0" objective: the model directly predicts the
        clean embeddings rather than predicting the noise. This is equivalent
        mathematically but empirically works better for discrete diffusion
        (Li et al., 2022 — Diffusion-LM).
        """
        batch_size = token_ids.size(0)

        # Step 1: Get clean embeddings for the input tokens
        x_0 = self.get_clean_embeddings(token_ids)  # (batch, seq_len, embed_dim)

        # Step 2: Sample a random timestep for each sequence in the batch
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=token_ids.device)

        # Step 3: Add noise to get x_t
        x_t, _ = self.forward_diffusion(x_0, t)

        # Step 4: Predict the clean embeddings from the noisy input
        x_0_pred = self.denoiser(x_t, t)

        # Step 5: MSE loss between predicted and actual clean embeddings
        loss = F.mse_loss(x_0_pred, x_0)

        return loss

    @torch.no_grad()
    def generate(self, seq_len: int = 64, batch_size: int = 1) -> list[str]:
        """
        Generate text by running the reverse diffusion process.

        Starts from pure Gaussian noise and iteratively denoises to produce
        clean embeddings, then maps those to the nearest vocabulary tokens.

        Note: this is a simplified DDPM sampler. More advanced samplers
        (DDIM, etc.) can produce better results with fewer steps.
        """
        self.eval()
        device = next(self.parameters()).device

        # Start from pure noise — this is x_T
        x_t = torch.randn(batch_size, seq_len, self.embed_dim, device=device)

        # Reverse diffusion: denoise from t=T-1 down to t=0
        for t_val in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)

            # Model predicts the clean embeddings x_0
            x_0_pred = self.denoiser(x_t, t)

            if t_val > 0:
                # Compute x_{t-1} from x_t using the predicted x_0
                # This is the standard DDPM reverse step formula
                alpha_t = self.schedule.alphas[t_val]
                alpha_bar_t = self.schedule.alpha_bar[t_val]
                alpha_bar_prev = self.schedule.alpha_bar[t_val - 1]
                beta_t = self.schedule.betas[t_val]

                # Posterior mean: weighted combination of x_t and predicted x_0
                # mu_t = (sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t)) * x_0_pred
                #       + (sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)) * x_t
                coef_x0 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1.0 - alpha_bar_t)
                coef_xt = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t)
                mean = coef_x0 * x_0_pred + coef_xt * x_t

                # Posterior variance — use the "fixed small" variance for simplicity
                variance = (beta_t * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t)
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(variance) * noise
            else:
                # At t=0, just use the predicted clean embeddings directly
                x_t = x_0_pred

        # Map continuous embeddings back to discrete tokens via nearest-neighbor
        # lookup in the embedding table (cosine similarity)
        embed_weight = self.denoiser.token_embed.weight  # (vocab_size, embed_dim)
        # Normalize both for cosine similarity
        x_norm = F.normalize(x_t, dim=-1)                # (batch, seq_len, embed_dim)
        w_norm = F.normalize(embed_weight, dim=-1)        # (vocab_size, embed_dim)
        # Cosine similarity: (batch, seq_len, vocab_size)
        similarity = torch.matmul(x_norm, w_norm.T)
        # Pick the token with highest similarity at each position
        token_ids = similarity.argmax(dim=-1)             # (batch, seq_len)

        return token_ids.cpu().tolist()


# ===========================================================================
# 5. TRAINING LOOP
# ===========================================================================

def train(
    model: DiffusionLM,
    dataset: ShakespeareDataset,
    tokenizer: CharTokenizer,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    log_every: int = 100,
    sample_every: int = 500,
):
    """
    Standard training loop with AdamW optimizer and cosine LR decay.
    Periodically generates samples so you can watch the model learn.
    """
    device = next(model.parameters()).device

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Cosine annealing schedule — decays LR smoothly over training
    total_steps = epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    global_step = 0
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(device)

            # Forward pass: compute diffusion training loss
            loss = model.compute_loss(batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Log training progress
            if global_step % log_every == 0:
                avg_loss = epoch_loss / num_batches
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"  step {global_step:>6d} | "
                    f"loss {loss.item():.4f} | "
                    f"avg {avg_loss:.4f} | "
                    f"lr {current_lr:.2e}"
                )

            # Generate a sample to see how the model is progressing
            if global_step % sample_every == 0:
                print("\n--- Sample generation ---")
                token_ids_list = model.generate(seq_len=64, batch_size=2)
                for i, ids in enumerate(token_ids_list):
                    text = tokenizer.decode(ids)
                    print(f"  Sample {i+1}: {repr(text)}")
                print("--- End samples ---\n")
                model.train()  # switch back to training mode after generate()

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{epochs} complete — avg loss: {avg_epoch_loss:.4f}")

    return model


# ===========================================================================
# 6. MAIN — put it all together
# ===========================================================================

def main():
    # --- Hyperparameters ---
    SEQ_LEN = 128       # characters per training chunk
    EMBED_DIM = 256     # embedding / model hidden dimension
    NUM_HEADS = 8       # attention heads (256 / 8 = 32 dim per head)
    NUM_LAYERS = 4      # transformer encoder layers
    FF_DIM = 1024       # feedforward inner dimension (4x embed_dim)
    NUM_TIMESTEPS = 1000  # diffusion steps
    DROPOUT = 0.1

    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4

    # --- Load data ---
    print("Loading Shakespeare corpus...")
    text = load_shakespeare_text(SHAKESPEARE_CSV)
    print(f"  Corpus length: {len(text):,} characters")

    # --- Build tokenizer ---
    tokenizer = CharTokenizer(text)
    print(f"  Vocabulary size: {tokenizer.vocab_size} (characters + pad)")

    # --- Tokenize and build dataset ---
    token_ids = tokenizer.encode(text)
    dataset = ShakespeareDataset(token_ids, seq_len=SEQ_LEN)
    print(f"  Training chunks: {len(dataset):,} sequences of {SEQ_LEN} chars")

    # --- Build model ---
    model = DiffusionLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        max_seq_len=SEQ_LEN,
        num_timesteps=NUM_TIMESTEPS,
        dropout=DROPOUT,
    ).to(DEVICE)

    # Print model size for reference
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # --- Train ---
    print("\nStarting training...\n")
    model = train(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        log_every=100,
        sample_every=500,
    )

    # --- Final generation ---
    print("\n" + "=" * 60)
    print("FINAL GENERATION — 5 samples of 128 characters each:")
    print("=" * 60)
    token_ids_list = model.generate(seq_len=128, batch_size=5)
    for i, ids in enumerate(token_ids_list):
        text = tokenizer.decode(ids)
        print(f"\nSample {i+1}:")
        print(text)

    # --- Save model checkpoint ---
    save_path = os.path.join(os.path.dirname(__file__), "diffusion_lm_checkpoint.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "char_to_id": tokenizer.char_to_id,
            "id_to_char": tokenizer.id_to_char,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "ff_dim": FF_DIM,
            "seq_len": SEQ_LEN,
            "num_timesteps": NUM_TIMESTEPS,
        },
        save_path,
    )
    print(f"\nCheckpoint saved to {save_path}")


if __name__ == "__main__":
    main()
