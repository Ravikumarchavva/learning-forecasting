import os
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F

# =====================================================================
# 1. Hyperparameters & Configuration (~10.8 Million Parameters)
# Optimized for high-performance training on a 4GB GPU (RTX 3050 Laptop)
# =====================================================================
batch_size = 32        # Sequence batches per iteration
block_size = 192       # Context length (lookback window)
max_iters = 8000       # Total training iterations
eval_interval = 500    # Print evaluation loss and text sample
learning_rate = 3e-4   # AdamW learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 384           # Embedding dimension
n_head = 6             # Attention heads (384 / 6 = 64 dim per head)
n_layer = 8            # Transformer blocks (increased from 6 for larger capacity)
dropout = 0.2          # Regularization
# =====================================================================

dataset_path = "input_code_500k.txt"

def load_data():
    if os.path.exists(dataset_path):
        print(f"Loading formatted dataset from local cache: '{dataset_path}'")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        from datasets import load_dataset
        print("Loading python-code-dataset-500k from Hugging Face...")
        # Load the train split
        ds = load_dataset('jtatman/python-code-dataset-500k', split='train')
        
        text_pieces = []
        # Compile a 5,000-example subset to fit on a 4GB GPU laptop
        limit = min(5000, len(ds))
        for i in range(limit):
            row = ds[i]
            instruction = row['instruction']
            output = row['output']
            
            # Format instruction and output clearly
            text_pieces.append(f"USER: {instruction}\nASSISTANT: Here is the code:\n{output}\n")
            text_pieces.append("\n" + "="*40 + "\n\n")
            
        text = "".join(text_pieces)
        with open(dataset_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Dataset compiled and cached locally to: '{dataset_path}'")
    return text

# Load text
text = load_data()

# Set up custom Byte-Level BPE Tokenizer (similar to GPT-2/GPT-4)
tokenizer_file = "tokenizer_bpe.json"
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

if os.path.exists(tokenizer_file):
    print(f"Loading trained BPE Tokenizer from '{tokenizer_file}'...")
    tokenizer = Tokenizer.from_file(tokenizer_file)
else:
    print("Training custom Byte-Level BPE Tokenizer on dataset...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], vocab_size=8000)
    tokenizer.train([dataset_path], trainer)
    tokenizer.save(tokenizer_file)
    print(f"BPE Tokenizer trained and saved to: '{tokenizer_file}'")

vocab_size = tokenizer.get_vocab_size()

# Define BPE encode/decode functions
encode = lambda s: tokenizer.encode(s).ids
decode = lambda l: tokenizer.decode(l)

# Convert dataset to PyTorch tensors
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # Train/Val Split (90/10)
train_data = data[:n]
val_data = data[n:]

# Data Loader
def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# =====================================================================
# 2. Transformer Decoder Architecture
# =====================================================================
class MultiHeadAttention(nn.Module):
    """ Causal Multi-Head Attention using PyTorch 2.0+ scaled_dot_product_attention (FlashAttention) """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        
        # Projections for Key, Query, Value
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        
        # Project and reshape to: (B, T, nh, hs) -> (B, nh, T, hs)
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        
        # PyTorch 2.0 fast native Scaled Dot Product Attention (FlashAttention)
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=dropout if self.training else 0.0, 
            is_causal=True
        )
        
        # Reshape and project back: (B, nh, T, hs) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ MLP Block """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Causal Transformer Block """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    """ Generative Pretrained Transformer (GPT) """
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        x = self.blocks(x)    # (B, T, n_embd)
        x = self.ln_f(x)      # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# =====================================================================
# 3. Training & Generation Execution Modes
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Train or generate text using a 20M Parameter GPT Model.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help="Train the model on the Python Code dataset.")
    group.add_argument('--generate', type=str, help="Generate text starting with the given prompt.")
    
    args = parser.parse_args()
    
    model = GPTLanguageModel().to(device)
    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}")
    print(f"Trainable model parameters: {params_count:,}")
    
    if args.train:
        print("Training mode selected. Starting loop...")
        weights_path = 'model.pth'
        if os.path.exists(weights_path):
            print(f"Found existing weights at '{weights_path}'. Resuming training...")
            model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            print("No existing weights found. Training from scratch...")
            
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        for iter in range(max_iters):
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss(model)
                print(f"Step {iter:4d}: Train Loss {losses['train']:.4f} | Val Loss {losses['val']:.4f}")
                
                # Check sample output
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                sample = decode(model.generate(context, max_new_tokens=100)[0].tolist())
                print("--- Mid-Train Generation ---")
                print(sample)
                print("----------------------------\n")
                
            xb, yb = get_batch('train')
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
        torch.save(model.state_dict(), 'model.pth')
        print("Model weights saved successfully to model.pth!")
        
    elif args.generate:
        print(f"Generation mode selected. Prompt: '{args.generate}'")
        weights_path = 'model.pth'
        if not os.path.exists(weights_path):
            print("Error: Could not find trained weights at 'model.pth'. Please train the model first.")
            return
            
        # Load weights
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        
        # Encode prompt
        encoded_prompt = encode(args.generate)
        if len(encoded_prompt) == 0:
            encoded_prompt = [0]
            
        context = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
        print("\n--- Final Generation Output ---")
        print(decode(model.generate(context, max_new_tokens=400)[0].tolist()))
        print("--------------------------------")

if __name__ == '__main__':
    main()
