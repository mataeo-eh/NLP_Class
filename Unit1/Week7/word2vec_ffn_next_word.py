"""
Next-Word Prediction (Neural n-gram LM) with Word2Vec + Feedforward Neural Network

Predict the next word given the previous N words:
    (w_{t-N}, ..., w_{t-2}, w_{t-1}) -> w_t

This script:
1) Trains a BPE tokenizer on a plain-text corpus (or loads one from a .pkl file)
2) Trains Word2Vec embeddings with gensim on BPE-tokenized sentences
3) Trains a fully-connected FFN in PyTorch on concatenated Word2Vec embeddings
4) Saves all artifacts (tokenizer, W2V, FFN weights, vocab) to an output directory
5) Prints a few top-k next-word predictions using the trained model
6) Optionally evaluates perplexity across all saved models using a cloze test

Requirements:
    pip install gensim torch numpy tokenizers tqdm

Notes:
- Fixed-window n-gram model; context size is set with --context_size.
- BPE tokenizer handles punctuation and supports subword OOV decomposition.
- Word2Vec embeddings are trained on BPE tokens and used as fixed FFN features.
- All models in --output share a perplexity.csv when --perplexity is run.

────────────────────────────────────────────────────────────────────────────────
Usage Examples
────────────────────────────────────────────────────────────────────────────────

# Train a new BPE tokenizer and LM from scratch.
# The tokenizer is trained on --training_corpus_path and saved to --output.
# The LM (W2V + FFN) is then trained on --corpus_path.
python Unit1/Week7/word2vec_ffn_next_word.py \
  --training_corpus_path Unit1/Week7/language_model_corpus_50000_sentences.txt \
  --corpus_path Unit1/Week7/language_model_corpus_50000_sentences.txt \
  --output Unit1/Week7 \
  --name 2gram \
  --context_size 2

# Train a different n-gram model reusing an already-trained tokenizer.
# Skips tokenizer training entirely and goes straight to LM training.
python Unit1/Week7/word2vec_ffn_next_word.py \
  --load_token_dict Unit1/Week7/2gram_tokenizer.pkl \
  --corpus_path Unit1/Week7/language_model_corpus_50000_sentences.txt \
  --output Unit1/Week7 \
  --name 3gram \
  --context_size 3

# Retrain just the LM while keeping the existing tokenizer.
# Delete the _ffn.pt, _w2v.model, and _vocab.pkl files for the model name,
# then rerun — the tokenizer pkl will be detected and skipped automatically.
python Unit1/Week7/word2vec_ffn_next_word.py \
  --load_token_dict Unit1/Week7/2gram_tokenizer.pkl \
  --corpus_path Unit1/Week7/language_model_corpus_50000_sentences.txt \
  --output Unit1/Week7 \
  --name 2gram \
  --context_size 2

# Load all artifacts from disk and run demo predictions only.
# All three LM files exist so training is skipped entirely.
python Unit1/Week7/word2vec_ffn_next_word.py \
  --load_token_dict Unit1/Week7/2gram_tokenizer.pkl \
  --output Unit1/Week7 \
  --name 2gram \
  --context_size 2

# Run perplexity evaluation across every *_ffn.pt model in --output.
# Scores all models found in the directory and writes a single perplexity.csv.
# Each model's paired *_tokenizer.pkl is discovered automatically — no --load_token_dict needed.
python Unit1/Week7/word2vec_ffn_next_word.py \
  --output Unit1/Week7 \
  --test_corpus Unit1/Week7/cloze_corpus_500_sentences.txt \
  --answers Unit1/Week7/cloze_corpus_500_answers.txt \
  --perplexity

# Alternatively, pass --load_token_dict to also run demo predictions for a specific model.
python Unit1/Week7/word2vec_ffn_next_word.py \
  --load_token_dict Unit1/Week7/2gram_tokenizer.pkl \
  --output Unit1/Week7 \
  --name 2gram \
  --context_size 2 \
  --test_corpus Unit1/Week7/cloze_corpus_500_sentences.txt \
  --answers Unit1/Week7/cloze_corpus_500_answers.txt \
  --perplexity

# Train a new model and immediately evaluate perplexity in one shot.
# Useful when you want a full train+eval run without a second command.
python Unit1/Week7/word2vec_ffn_next_word.py \
  --training_corpus_path Unit1/Week7/language_model_corpus_50000_sentences.txt \
  --corpus_path Unit1/Week7/language_model_corpus_50000_sentences.txt \
  --output Unit1/Week7 \
  --name 4gram \
  --context_size 4 \
  --test_corpus Unit1/Week7/cloze_corpus_500_sentences.txt \
  --answers Unit1/Week7/cloze_corpus_500_answers.txt \
  --perplexity
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import copy
import csv
import glob
import math
import os
import re
import random
from dataclasses import dataclass
from typing import List, Tuple

import pickle

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from gensim.models import Word2Vec
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, Whitespace, Punctuation



# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    seed: int = 42
    context_size: int = 3           # previous 3 words
    w2v_vector_size: int = 150      # richer embeddings for subword token space
    w2v_window: int = 10            # wider context for subword co-occurrence
    w2v_min_count: int = 1
    w2v_sg: int = 1                 # 1 = skip-gram, 0 = CBOW
    w2v_epochs: int = 200           # upper bound; early stopping exits before this if loss stalls
    w2v_patience: int = 5           # W2V early-stop: consecutive epochs with no loss improvement
    w2v_negative: int = 10          # negative sampling; higher = richer gradients

    hidden_dim: int = 128
    dropout: float = 0.1
    batch_size: int = 512
    lr: float = 1e-3
    epochs: int = 60
    num_workers: int = 0        # set to 2-4 with a real corpus to overlap CPU data prep with MPS
    patience: int = 5           # FFN early-stop: consecutive epochs where val rises + train falls
    corpus_path: str | None = None  # path to a plain-text corpus file; None uses the built-in toy corpus


CFG = Config()


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class Tokenizer:
    """BPE tokenizer backed by HuggingFace tokenizers.

    Vocabulary size is not hardcoded — the effective vocabulary is driven by
    corpus statistics via min_frequency.  vocab_size is set to a generous
    ceiling so the trainer stops on data exhaustion, not an arbitrary cap.

    Pre-tokenization uses Whitespace followed by Punctuation so that
    punctuation marks are separated from adjacent words before BPE merges,
    giving the model subword tokens that cross word/punctuation boundaries
    only intentionally.
    """

    _VOCAB_CEIL = 100_000  # generous upper bound; real vocab is data-driven

    def __init__(self, corpus_path: str | None = None, output_path: str | None = None):
        self.corpus_path = corpus_path
        self.output_path = output_path
        self._tokenizer: HFTokenizer | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self) -> "Tokenizer":
        """Train BPE on self.corpus_path and pickle the result to self.output_path."""
        if not self.corpus_path:
            raise ValueError("corpus_path must be set before calling train()")
        if not self.output_path:
            raise ValueError("output_path must be set before calling train()")

        hf_tok = HFTokenizer(BPE(unk_token="[UNK]"))
        # Split on whitespace first, then isolate punctuation so that commas,
        # periods, etc. become their own pre-token candidates for BPE merges.
        hf_tok.pre_tokenizer = Sequence([Whitespace(), Punctuation()])

        trainer = BpeTrainer(
            special_tokens=["[UNK]"],
            vocab_size=self._VOCAB_CEIL,
            min_frequency=1,
            show_progress=True,
        )

        hf_tok.train(files=[self.corpus_path], trainer=trainer)
        self._tokenizer = hf_tok

        with open(self.output_path, "wb") as fh:
            pickle.dump(hf_tok, fh)

        print(
            f"BPE tokenizer trained. Vocab size: {hf_tok.get_vocab_size():,}. "
            f"Saved to '{self.output_path}'"
        )
        return self

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        """Load a pickled HFTokenizer from path."""
        with open(path, "rb") as fh:
            hf_tok = pickle.load(fh)
        instance = cls()
        instance._tokenizer = hf_tok
        print(f"BPE tokenizer loaded from '{path}'. Vocab size: {hf_tok.get_vocab_size():,}")
        return instance

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[str]:
        """Return a list of subword token strings."""
        return self._tokenizer.encode(text).tokens

    # ------------------------------------------------------------------
    # Vocab
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()




def split_sentences(raw_text: str, tokenizer: Tokenizer) -> List[List[str]]:
    """Split into sentences, then BPE-tokenize each for Word2Vec training."""
    chunks = [s for s in re.split(r"[.!?]+", raw_text) if s.strip()]
    return [
        tokenizer.encode(s)
        for s in tqdm(chunks, desc="Tokenizing sentences", unit="sent", leave=False)
    ]


def build_4gram_pairs(tokens: List[str], context_size: int) -> List[Tuple[List[str], str]]:
    """Build (context -> target) pairs with context_size previous tokens."""
    pairs = []
    for i in tqdm(range(context_size, len(tokens)), desc="Building n-gram pairs", unit="pair", leave=False):
        context = tokens[i - context_size:i]
        target = tokens[i]
        pairs.append((context, target))
    return pairs


def context_to_x(w2v: Word2Vec, context_words: List[str]) -> np.ndarray:
    """Concatenate Word2Vec embeddings for the context words."""
    vecs = [w2v.wv[w] for w in context_words]
    return np.concatenate(vecs).astype(np.float32)


# -----------------------------
# Dataset
# -----------------------------
class NgramIterableDataset(IterableDataset):
    """Lazy n-gram dataset.

    Stores raw (context_words, target_word) string pairs and computes
    Word2Vec embeddings on the fly inside __iter__, so the full float32
    feature matrix is never materialised in memory.

    Worker-safe splitting
    ---------------------
    With num_workers > 0, PyTorch runs each worker's __iter__ independently
    in a separate process. Without explicit splitting every worker would
    iterate every pair, feeding each sample num_workers times per epoch —
    silent data duplication that inflates gradient updates and causes the
    model to silently overfit.

    Fix: inside __iter__ we build a local index list, optionally shuffle it
    (local copy — no shared mutable state between workers), then slice it
    into num_workers contiguous, non-overlapping chunks using get_worker_info().
    Each worker only yields samples whose index falls in its assigned chunk.
    math.ceil ensures every pair is covered even when len(pairs) is not evenly
    divisible by num_workers; the last worker's slice is simply shorter.
    """

    def __init__(
        self,
        pairs: List[Tuple[List[str], str]],
        w2v: Word2Vec,
        word_to_idx: dict,
        shuffle: bool = False,
    ) -> None:
        self.pairs = pairs
        self.w2v = w2v
        self.word_to_idx = word_to_idx
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        # Build a fresh local index list every call (one per worker per epoch).
        # Shuffling here is safe: each worker operates on its own copy of
        # indices and never touches another worker's copy.
        indices = list(range(len(self.pairs)))
        if self.shuffle:
            random.shuffle(indices)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process dataloader: iterate all indices.
            worker_indices = indices
        else:
            # Multi-process: assign each worker a unique contiguous slice of
            # the (already shuffled) index list.  Contiguous slicing of a
            # pre-shuffled list is equivalent to a globally random assignment
            # while still guaranteeing zero overlap between workers.
            per_worker = math.ceil(len(indices) / worker_info.num_workers)
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(indices))
            worker_indices = indices[start:end]

        for i in worker_indices:
            context_words, target_word = self.pairs[i]
            x = context_to_x(self.w2v, context_words)
            y = self.word_to_idx[target_word]
            yield torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# -----------------------------
# Model
# -----------------------------
class FFNLanguageModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def accuracy_from_logits(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y_true).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str | None = None,
) -> Tuple[float, float]:
    train = optimizer is not None
    model.train(train)

    total_loss, total_acc, total_n = 0.0, 0.0, 0
    bar = tqdm(loader, desc=desc, leave=False, unit="batch", mininterval=0.5) if desc else loader
    for xb, yb in bar:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        acc = accuracy_from_logits(logits, yb)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        total_n += bs

    return total_loss / total_n, total_acc / total_n


@torch.no_grad()
def predict_next(
    model: nn.Module,
    w2v: Word2Vec,
    word_to_idx: dict,
    idx_to_word: dict,
    prev_words: List[str],
    topk: int = 5,
    device: torch.device | None = None,
) -> List[Tuple[str, float]]:
    """Return top-k next-word predictions and probabilities."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    prev_words = [w.lower() for w in prev_words]
    for w in prev_words:
        if w not in w2v.wv:
            raise ValueError(f"Word '{w}' not in Word2Vec vocabulary. Train on more text or change the input.")

    x = context_to_x(w2v, prev_words)
    xb = torch.from_numpy(x).unsqueeze(0).to(device)

    model.eval()
    logits = model(xb).squeeze(0)
    probs = torch.softmax(logits, dim=0).detach().cpu().numpy()

    top_idx = np.argsort(-probs)[:topk]
    return [(idx_to_word[i], float(probs[i])) for i in top_idx]


@torch.no_grad()
def compute_perplexity(
    model: FFNLanguageModel,
    w2v: Word2Vec,
    tokenizer: "Tokenizer",
    word_to_idx: dict,
    sentences: List[str],
    answers: List[str],
    context_size: int,
    device: torch.device,
) -> Tuple[float, int, dict]:
    """Compute perplexity over cloze-test sentence/answer pairs.

    PP = exp(-1/N * Σ log P(correct_answer | context))

    Returns (perplexity, n_evaluated, skip_counts) where skip_counts is a dict
    mapping each skip reason to its count.
    """
    model.eval()
    total_nll   = 0.0
    n_evaluated = 0
    skip_counts = {
        "no_blank":           0,  # sentence missing [BLANK]
        "context_too_short":  0,  # fewer tokens before [BLANK] than context_size
        "context_oov":        0,  # a context token not in Word2Vec vocab
        "empty_answer":       0,  # answer tokenized to nothing
        "answer_oov":         0,  # answer token not in model vocab
    }

    for sentence, answer in tqdm(
        zip(sentences, answers),
        total=len(sentences),
        desc="  Scoring",
        unit="sent",
        leave=False,
    ):
        # Tokenize the text before [BLANK] to get context
        parts = sentence.split("[BLANK]")
        if len(parts) < 2:
            skip_counts["no_blank"] += 1
            continue

        context_tokens = tokenizer.encode(parts[0].strip())
        if len(context_tokens) < context_size:
            skip_counts["context_too_short"] += 1
            continue

        context = context_tokens[-context_size:]
        if any(t not in w2v.wv for t in context):
            skip_counts["context_oov"] += 1
            continue

        # Tokenize answer; use the first subword token
        answer_tokens = tokenizer.encode(answer.strip())
        if not answer_tokens:
            skip_counts["empty_answer"] += 1
            continue

        answer_token = answer_tokens[0]
        if answer_token not in word_to_idx:
            skip_counts["answer_oov"] += 1
            continue

        answer_idx = word_to_idx[answer_token]

        x      = context_to_x(w2v, context)
        xb     = torch.from_numpy(x).unsqueeze(0).to(device)
        logits = model(xb).squeeze(0)
        probs  = torch.softmax(logits, dim=0)

        prob_correct = max(probs[answer_idx].item(), 1e-10)  # guard against log(0)
        total_nll   += -math.log(prob_correct)
        n_evaluated += 1

    if n_evaluated == 0:
        return float("inf"), 0, skip_counts

    return math.exp(total_nll / n_evaluated), n_evaluated, skip_counts


def run_perplexity_tests(
    output_dir: str,
    test_corpus_path: str,
    answers_path: str,
    device: torch.device,
    cfg: "Config",
) -> None:
    """Find every *_ffn.pt in output_dir, evaluate perplexity, save perplexity.csv."""
    # Load test data
    with open(test_corpus_path, "r", encoding="utf-8") as fh:
        sentences = [l.strip() for l in fh if l.strip()]
    with open(answers_path, "r", encoding="utf-8") as fh:
        answers = [l.strip() for l in fh if l.strip()]

    if len(sentences) != len(answers):
        raise ValueError(
            f"Sentence count ({len(sentences)}) != answer count ({len(answers)}). "
            "Each line in the answers file must correspond to a line in the sentences file."
        )
    print(f"Loaded {len(sentences)} cloze instances from '{test_corpus_path}'")

    ffn_files = sorted(glob.glob(os.path.join(output_dir, "*_ffn.pt")))
    if not ffn_files:
        raise SystemExit(f"No *_ffn.pt files found in '{output_dir}'")

    csv_path = os.path.join(output_dir, "perplexity.csv")
    results  = []

    for ffn_path in ffn_files:
        name       = os.path.basename(ffn_path)[: -len("_ffn.pt")]
        tok_path   = os.path.join(output_dir, f"{name}_tokenizer.pkl")
        w2v_path   = os.path.join(output_dir, f"{name}_w2v.model")
        vocab_path = os.path.join(output_dir, f"{name}_vocab.pkl")

        missing = [p for p in [tok_path, w2v_path, vocab_path] if not os.path.exists(p)]
        if missing:
            print(f"  Skipping '{name}': missing {[os.path.basename(p) for p in missing]}")
            continue

        print(f"\nEvaluating '{name}'...")

        model_tokenizer = Tokenizer.load(tok_path)

        with open(vocab_path, "rb") as fh:
            vocab_data  = pickle.load(fh)
        word_to_idx = vocab_data["word_to_idx"]
        vocab_size  = vocab_data["vocab_size"]

        w2v     = Word2Vec.load(w2v_path)
        emb_dim = w2v.vector_size

        # Derive context_size from the saved weight shapes — no CLI arg needed
        state          = torch.load(ffn_path, map_location="cpu", weights_only=True)
        first_layer_in = state["net.0.weight"].shape[1]
        context_size   = first_layer_in // emb_dim

        model = FFNLanguageModel(
            input_dim=context_size * emb_dim,
            hidden_dim=cfg.hidden_dim,
            vocab_size=vocab_size,
            dropout=0.0,  # no dropout at eval time
        ).to(device)
        model.load_state_dict(state)
        model.eval()

        perplexity, n_eval, skip_counts = compute_perplexity(
            model, w2v, model_tokenizer, word_to_idx,
            sentences, answers, context_size, device,
        )
        n_skip = sum(skip_counts.values())

        print(f"  Perplexity : {perplexity:.4f}")
        print(f"  Evaluated  : {n_eval} / {len(sentences)}  |  Skipped: {n_skip}")
        if n_skip:
            for reason, count in skip_counts.items():
                if count:
                    print(f"    {reason}: {count}")
        results.append((name, round(perplexity, 6), n_eval, n_skip, *skip_counts.values()))

    skip_reason_headers = ["skip_no_blank", "skip_context_too_short", "skip_context_oov",
                           "skip_empty_answer", "skip_answer_oov"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "perplexity", "n_evaluated", "n_skipped_total"] + skip_reason_headers)
        writer.writerows(results)

    print(f"\nPerplexity results saved to '{csv_path}'")


_TOY_CORPUS = """
In the beginning the world was quiet and the sky was clear.
The people in the village told stories by the fire at night.
One day a traveler arrived with a map and a mysterious key.
The traveler said the key would open a door beyond the hills.
Beyond the hills there was a forest and beyond the forest a river.
The river flowed into the sea and the sea met the sky.
The village listened and the village wondered what was true.
""".strip()


def load_corpus(path: str | None, context_size: int) -> str:
    """Return raw corpus text from a file path, or the built-in toy corpus.

    Errors caught:
        - None path          → silently returns the toy corpus
        - Directory given    → raises ValueError (not a file)
        - File not found     → raises FileNotFoundError with the path
        - Permission denied  → raises PermissionError with the path
        - Encoding issue     → tries UTF-8, falls back to latin-1; raises
                               ValueError only if both fail
        - Empty file         → raises ValueError
        - Too few tokens     → raises ValueError (need > context_size tokens
                               to form at least one training pair)
    """
    if path is None:
        print("corpus_path not set — using built-in toy corpus")
        return _TOY_CORPUS

    if os.path.isdir(path):
        raise ValueError(f"corpus_path points to a directory, not a file: '{path}'")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus file not found: '{path}'")

    if not os.access(path, os.R_OK):
        raise PermissionError(f"No read permission for corpus file: '{path}'")

    # Try UTF-8 first; fall back to latin-1 for files with legacy encodings.
    text: str | None = None
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as fh:
                text = fh.read()
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        raise ValueError(
            f"Could not decode '{path}' as UTF-8 or latin-1. "
            "Convert the file to UTF-8 and try again."
        )

    text = text.strip()
    if not text:
        raise ValueError(f"Corpus file is empty: '{path}'")

    token_count = len(re.findall(r"[a-z']+", text.lower()))
    if token_count <= context_size:
        raise ValueError(
            f"Corpus too short: found {token_count} token(s) in '{path}' but need "
            f"at least {context_size + 1} to form one training pair "
            f"(context_size={context_size})."
        )

    print(f"Loaded corpus from '{path}' ({os.path.getsize(path):,} bytes)")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Next-word prediction with Word2Vec + FFN")
    parser.add_argument("--context_size", type=int, default=None, help="Number of previous words used as context")
    parser.add_argument("--corpus_path", type=str, default=None, help="Path to a plain-text corpus file for LM training")
    parser.add_argument("--output", type=str, default=None, help="Directory to save all model artifacts (tokenizer, FFN, W2V, vocab)")
    parser.add_argument("--name", type=str, default="lm", help="Name prefix for saved model files (e.g. 'my_model' → my_model_ffn.pt, my_model_w2v.model, ...)")
    parser.add_argument("--training_corpus_path", type=str, default=None, help="Path to corpus for training the BPE tokenizer")
    parser.add_argument("--load_token_dict", type=str, default=None, help="Path to a pre-trained BPE tokenizer (.pkl) to load")
    parser.add_argument("--test_corpus", type=str, default=None, help="Path to cloze-test sentences file (use [BLANK] to mark the missing word)")
    parser.add_argument("--answers", type=str, default=None, help="Path to answers file (one answer per line, matching rows in --test_corpus)")
    parser.add_argument("--perplexity", "-p", action="store_true", help="Run perplexity evaluation on all *_ffn.pt models in --output and save perplexity.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.context_size is not None:
        CFG.context_size = args.context_size
    if args.corpus_path is not None:
        CFG.corpus_path = args.corpus_path

    set_seed(CFG.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    # Derive output directory and per-artifact file paths
    output_dir = args.output if args.output else "."
    name = args.name
    os.makedirs(output_dir, exist_ok=True)

    tok_path   = os.path.join(output_dir, f"{name}_tokenizer.pkl")
    ffn_path   = os.path.join(output_dir, f"{name}_ffn.pt")
    w2v_path   = os.path.join(output_dir, f"{name}_w2v.model")
    vocab_path = os.path.join(output_dir, f"{name}_vocab.pkl")

    # Perplexity-only mode: no tokenizer arg supplied, so skip training/loading
    # entirely and let run_perplexity_tests pair each *_ffn.pt with its own
    # *_tokenizer.pkl found in the same output directory.
    perplexity_only = (
        args.perplexity
        and args.training_corpus_path is None
        and args.load_token_dict is None
    )

    if perplexity_only:
        if not args.test_corpus or not args.answers:
            raise SystemExit("Error: --perplexity requires --test_corpus and --answers")
        run_perplexity_tests(
            output_dir=output_dir,
            test_corpus_path=args.test_corpus,
            answers_path=args.answers,
            device=device,
            cfg=CFG,
        )
        return

    # -----------------------------
    # 0) BPE Tokenizer
    # -----------------------------
    if args.training_corpus_path is not None:
        if os.path.exists(tok_path):
            print(f"... tokenizer skipped ('{tok_path}' already exists) ...")
            bpe_tokenizer = Tokenizer.load(tok_path)
        else:
            bpe_tokenizer = Tokenizer(
                corpus_path=args.training_corpus_path,
                output_path=tok_path,
            ).train()
    elif args.load_token_dict is not None:
        bpe_tokenizer = Tokenizer.load(args.load_token_dict)
    else:
        raise SystemExit(
            "Error: provide either --training_corpus_path (to train a BPE tokenizer) "
            "or --load_token_dict (to load one from a .pkl file)."
        )

    # -----------------------------
    # LM: load saved artifacts or train from scratch
    # -----------------------------
    lm_exists = all(os.path.exists(p) for p in [ffn_path, w2v_path, vocab_path])

    if lm_exists:
        print(f"... LM skipped ('{name}' artifacts already exist in '{output_dir}') ...")
        with open(vocab_path, "rb") as fh:
            vocab_data = pickle.load(fh)
        word_to_idx = vocab_data["word_to_idx"]
        idx_to_word = vocab_data["idx_to_word"]
        vocab_size  = vocab_data["vocab_size"]

        w2v = Word2Vec.load(w2v_path)
        emb_dim = w2v.vector_size

        model = FFNLanguageModel(
            input_dim=CFG.context_size * emb_dim,
            hidden_dim=CFG.hidden_dim,
            vocab_size=vocab_size,
            dropout=CFG.dropout,
        ).to(device)
        model.load_state_dict(torch.load(ffn_path, map_location=device, weights_only=True))
        model.eval()
        print(f"Loaded: FFN ← '{ffn_path}', W2V ← '{w2v_path}', vocab ← '{vocab_path}'")

    else:
        # -----------------------------
        # 1) Corpus
        # -----------------------------
        raw_text = load_corpus(CFG.corpus_path, CFG.context_size)

        sentences = split_sentences(raw_text, bpe_tokenizer)
        tokens    = [tok for sentence in sentences for tok in sentence]
        pairs     = build_4gram_pairs(tokens, CFG.context_size)

        print("Num tokens:", len(tokens))
        print("Num pairs:", len(pairs))
        print("Example pair:", pairs[0])

        # -----------------------------
        # 2) Train Word2Vec
        # -----------------------------
        w2v = Word2Vec(
            vector_size=CFG.w2v_vector_size,
            window=CFG.w2v_window,
            min_count=CFG.w2v_min_count,
            workers=1,  # deterministic
            sg=CFG.w2v_sg,
            seed=CFG.seed,
            negative=CFG.w2v_negative,
        )
        w2v.build_vocab(sentences)

        best_w2v_loss = float("inf")
        w2v_stall     = 0
        w2v_bar = tqdm(range(1, CFG.w2v_epochs + 1), desc="W2V Training", unit="epoch")
        for w2v_epoch in w2v_bar:
            w2v.train(
                sentences,
                total_examples=w2v.corpus_count,
                epochs=1,
                compute_loss=True,
            )
            epoch_loss = w2v.get_latest_training_loss()

            if epoch_loss < best_w2v_loss:
                best_w2v_loss = epoch_loss
                w2v_stall     = 0
            else:
                w2v_stall += 1

            w2v_bar.set_postfix(loss=f"{epoch_loss:.0f}", stall=f"{w2v_stall}/{CFG.w2v_patience}")

            if w2v_stall >= CFG.w2v_patience:
                tqdm.write(f"W2V early stopped at epoch {w2v_epoch} (loss stalled for {w2v_stall} consecutive epochs)")
                break
        else:
            tqdm.write(f"W2V training complete ({CFG.w2v_epochs} epochs)")

        vocab       = sorted(list(w2v.wv.index_to_key))
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        idx_to_word = {i: w for w, i in word_to_idx.items()}
        print("Vocab size:", len(vocab))
        print("Embedding dim:", w2v.vector_size)

        # -----------------------------
        # 3) Train/val split on raw pairs (no numpy arrays)
        # -----------------------------
        # Shuffle the full pair list once before splitting so the val set is a
        # random sample.  Per-epoch training shuffle is handled inside the dataset.
        random.shuffle(pairs)
        split = int(0.8 * len(pairs))
        train_pairs, val_pairs = pairs[:split], pairs[split:]

        train_ds = NgramIterableDataset(train_pairs, w2v, word_to_idx, shuffle=True)
        val_ds   = NgramIterableDataset(val_pairs,   w2v, word_to_idx, shuffle=False)

        # num_workers > 0: on macOS, spawn (not fork) is used, so each worker receives a
        # pickled copy of the dataset object (pairs strings + w2v model).  The w2v model
        # can be large; keep num_workers=0 on memory-constrained machines.
        # shuffle=False: IterableDataset raises an error if shuffle=True is passed to
        # DataLoader.  Per-epoch randomisation is handled inside NgramIterableDataset.__iter__.
        train_loader = DataLoader(
            train_ds, batch_size=CFG.batch_size, shuffle=False,
            num_workers=CFG.num_workers, persistent_workers=CFG.num_workers > 0,
            pin_memory=False,  # pin_memory is CUDA-only; False is correct for MPS
        )
        val_loader = DataLoader(
            val_ds, batch_size=CFG.batch_size, shuffle=False,
            num_workers=CFG.num_workers, persistent_workers=CFG.num_workers > 0,
            pin_memory=False,
        )

        # -----------------------------
        # 4) Model + Train
        # -----------------------------
        emb_dim = w2v.vector_size
        model = FFNLanguageModel(
            input_dim=CFG.context_size * emb_dim,
            hidden_dim=CFG.hidden_dim,
            vocab_size=len(vocab),
            dropout=CFG.dropout,
        ).to(device)

        loss_fn   = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel params: {num_params:,}")
        print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
        print(f"Training for up to {CFG.epochs} epochs — early stop patience={CFG.patience}\n")
        print(f"{'Epoch':>5} | {'Tr Loss':>8} {'Tr Acc':>7} | {'Va Loss':>8} {'Va Acc':>7} | {'Δ Va Loss':>10} | Status")
        print("-" * 75)

        best_val_loss   = float("inf")
        best_checkpoint = None
        best_epoch      = 0
        prev_val_loss   = float("inf")
        prev_tr_loss    = float("inf")
        stall_count     = 0
        diverge_count   = 0  # consecutive epochs where val loss rises AND train loss falls
        stagnant_count  = 0  # consecutive epochs where both losses barely move
        stopped_early   = False
        _stagnation_thr = 1e-4  # minimum meaningful loss change

        epoch_bar = tqdm(range(1, CFG.epochs + 1), desc="FFN Training", unit="epoch")
        for epoch in epoch_bar:
            tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, optimizer, device)
            va_loss, va_acc = run_epoch(model, val_loader,   loss_fn, None,      device)

            epoch_bar.set_postfix(
                tr_loss=f"{tr_loss:.4f}",
                va_loss=f"{va_loss:.4f}",
                va_acc=f"{va_acc:.3f}",
            )

            # Checkpoint whenever val loss improves
            val_improved = va_loss < best_val_loss
            if val_improved:
                best_val_loss   = va_loss
                best_checkpoint = copy.deepcopy(model.state_dict())
                best_epoch      = epoch
                stall_count     = 0
                diverge_count   = 0
            else:
                stall_count += 1

            # Divergence: val loss rose AND train loss fell this epoch
            if va_loss > prev_val_loss and tr_loss < prev_tr_loss:
                diverge_count += 1
            else:
                diverge_count = 0  # non-diverging epoch resets the streak

            # Stagnation: both losses barely moving — model has converged
            if (abs(va_loss - prev_val_loss) < _stagnation_thr and
                    abs(tr_loss - prev_tr_loss) < _stagnation_thr):
                stagnant_count += 1
            else:
                stagnant_count = 0

            overfit_gap = va_loss - tr_loss
            delta       = va_loss - prev_val_loss
            stopping    = diverge_count >= CFG.patience or stagnant_count >= CFG.patience

            if epoch % 10 == 0 or epoch == 1 or stopping:
                if val_improved:
                    status = "best"
                elif stopping:
                    if stagnant_count >= CFG.patience:
                        status = f"EARLY STOP (stagnant x{stagnant_count})"
                    else:
                        status = f"EARLY STOP (diverged x{diverge_count})"
                elif stall_count >= 5:
                    status = f"stalled ({stall_count})"
                else:
                    status = "ok"

                if overfit_gap > 0.5:
                    status += " [OVERFIT?]"

                delta_str = f"{delta:+.4f}" if epoch > 1 else "   —"
                tqdm.write(f"{epoch:>5} | {tr_loss:>8.4f} {tr_acc:>7.3f} | {va_loss:>8.4f} {va_acc:>7.3f} | {delta_str:>10} | {status}")

            prev_val_loss = va_loss
            prev_tr_loss  = tr_loss

            if stopping:
                if stagnant_count >= CFG.patience:
                    tqdm.write(f"\nEarly stop at epoch {epoch}: both losses stagnant for {stagnant_count} consecutive epochs (threshold={_stagnation_thr}).")
                else:
                    tqdm.write(f"\nEarly stop at epoch {epoch}: val loss rose while train loss fell for {diverge_count} consecutive epochs.")
                tqdm.write(f"Restoring weights from epoch {best_epoch} (best val loss: {best_val_loss:.4f})")
                model.load_state_dict(best_checkpoint)
                stopped_early = True
                break

        tqdm.write("-" * 75)
        if stopped_early:
            tqdm.write(f"Stopped early at epoch {epoch} of {CFG.epochs}. Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
        else:
            tqdm.write(f"Training complete ({CFG.epochs} epochs). Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")

        # -----------------------------
        # Save LM artifacts
        # -----------------------------
        torch.save(model.state_dict(), ffn_path)
        w2v.save(w2v_path)
        with open(vocab_path, "wb") as fh:
            pickle.dump({
                "word_to_idx": word_to_idx,
                "idx_to_word": idx_to_word,
                "vocab_size":  len(vocab),
            }, fh)
        print(f"Saved: FFN → '{ffn_path}', W2V → '{w2v_path}', vocab → '{vocab_path}'")

    # -----------------------------
    # 5) Demo predictions
    # -----------------------------
    vocab_tokens = list(word_to_idx.keys())
    rng = random.Random(CFG.seed)
    tests = [rng.sample(vocab_tokens, CFG.context_size) for _ in range(4)]

    print("\nTop-k next-word predictions:")
    for ctx in tests:
        preds = predict_next(model, w2v, word_to_idx, idx_to_word, ctx, topk=5, device=device)
        print(f"{ctx} -> {preds}")

    # -----------------------------
    # 6) Perplexity evaluation
    # -----------------------------
    if args.perplexity:
        if not args.test_corpus or not args.answers:
            raise SystemExit("Error: --perplexity requires --test_corpus and --answers")
        run_perplexity_tests(
            output_dir=output_dir,
            test_corpus_path=args.test_corpus,
            answers_path=args.answers,
            device=device,
            cfg=CFG,
        )


if __name__ == "__main__":
    main()
