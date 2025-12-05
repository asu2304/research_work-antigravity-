import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
import numpy as np

# Check if MPS (Metal) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else: 
    device = torch.device("cpu")
print("Using device:", device)

# 1. Data Creation (ABCDEABCDE... pattern, with permutations) ie 5 cycle repeat
class ABCAlternatingPatternDataset(Dataset):
    def __init__(self, combinations, context_len=20, vocab_size=26):
        
        super().__init__()
        self.vocab_size = vocab_size
        self.context_len = context_len
        
        self.data, self.labels = [], []
        for (l1, l2, l3, l4, l5) in combinations:
            pattern = [l1, l2, l3, l4, l5] * ((context_len) // 4)
            seq = pattern[:context_len + 1]
            x = [ord(c) - 65 for c in seq[:context_len]]
            y = [ord(c) - 65 for c in seq[1:context_len + 1]]
            self.data.append(x)
            self.labels.append(y)
            
    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
        
    def __len__(self):
        return len(self.data)

# 1.1 Train/Test Split 
def make_train_test_sets(num_train=600, num_test=100, context_len=20, vocab_size=26):
    alphabet = [chr(65 + i) for i in range(vocab_size)]
    all_combs = []
    for i in range(vocab_size):
        for j in range(vocab_size):
            for k in range(vocab_size):
                for l in range(vocab_size):
                    for m in range(vocab_size):
                        all_combs.append((alphabet[i], alphabet[j], alphabet[k], alphabet[l], alphabet[m]))
                        
    # Shuffle and sample for train/test
    # Setting seed for reproducibility
    random.seed(42)
    random.shuffle(all_combs)
    train_combinations = all_combs[:num_train]
    test_combinations  = all_combs[num_train : num_train + num_test]
    train_set = ABCAlternatingPatternDataset(train_combinations, context_len, vocab_size)
    test_set  = ABCAlternatingPatternDataset(test_combinations,  context_len, vocab_size)
    
    return train_set, test_set

# 2. Custom Transformer Block
class ManualAttention(nn.Module):
    def __init__(self, embed_dim, vocab_size=26, num_heads=1, remove_bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.Wqk = nn.Parameter(torch.empty(num_heads, embed_dim, embed_dim))
        
        # only for one head!!!
        self.Wov = nn.Parameter(torch.empty(num_heads, embed_dim, vocab_size))
        nn.init.normal_(self.Wqk, mean=0, std=0.1)
        nn.init.normal_(self.Wov, mean=0, std=0.1)

        if not remove_bias:
            self.b_Wo = nn.Parameter(torch.zeros(embed_dim))
        else:
            self.register_parameter('b_Wo', None)
            
    def forward(self, x, mask=None):
        # x: [B, T, D]
        B, T, D = x.shape
        
        # We assume num_heads=1 for simplicity in manual logic matching
        i = 0 
        x_wqk = torch.matmul(x, self.Wqk[i]) # [B, T, D]
        QK = torch.matmul(x_wqk, x.transpose(-2, -1)) # [B, T, T]
        
        scores = QK / (self.head_dim ** 0.5)
        if mask is not None:
             scores = scores.masked_fill(mask == 1, float('-1e9'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Output computation
        out = attn_weights @ x @ self.Wov[i] # [B, T, vocab_size]
        
        if self.b_Wo is not None:
            out = out + self.b_Wo
        return out

# 3. Simple Transformer Model
class SimpleTransformerScratch(nn.Module):
    def __init__(self,
                 vocab_size=26,
                 context_len=20,
                 embed_dim=26,
                 num_heads=1,
                 use_mlp=False,
                 mlp_hidden_dim=32,
                 use_identity_embedding=True,
                 use_identity_unembedding=True,
                 use_positional_encoding=False,
                 use_layernorm=False,
                 remove_bias=True):
        
        super().__init__()
        self.context_len = context_len
        self.embed_dim = embed_dim

        self.token_embedding_dim = embed_dim
        self.input_dim = vocab_size 
        
        if use_positional_encoding:
            self.pos_embedding = torch.eye(context_len).unsqueeze(0)
            self.register_buffer("pos_embedding_fixed", self.pos_embedding)
            self.positional_dim = context_len
        else:
            self.register_parameter("pos_embedding", None)
            self.positional_dim = 0
        
        if use_identity_embedding:
            self.embedding = nn.Embedding(vocab_size, vocab_size)
            with torch.no_grad():
                self.embedding.weight.copy_(torch.eye(vocab_size))
            self.embedding.weight.requires_grad = False
            embed_dim = vocab_size
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.use_positional_encoding = use_positional_encoding
        
        self.concat_dim = self.token_embedding_dim + self.positional_dim
        self.attn_block = ManualAttention(embed_dim=self.concat_dim, vocab_size=vocab_size, num_heads=num_heads, remove_bias=remove_bias)

        if use_mlp:
            self.ff1 = nn.Linear(self.concat_dim, mlp_hidden_dim, bias=not remove_bias)
            self.ff2 = nn.Linear(mlp_hidden_dim, self.concat_dim, bias=not remove_bias)
            self.act = nn.ReLU()
        else:
            self.ff1 = nn.Identity()
            self.ff2 = nn.Identity()
            self.act = nn.Identity()
        if use_layernorm:
            self.ln1 = nn.LayerNorm(self.concat_dim)
            self.ln2 = nn.LayerNorm(self.concat_dim) if use_mlp else nn.Identity()
        else:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()
        if use_identity_unembedding: 
            self.output_proj = nn.Identity()
        else:
            self.output_proj = nn.Linear(self.concat_dim, vocab_size, bias=not remove_bias)
            
    # no residual connenctions
    def forward(self, x):
        emb = self.embedding(x)
        if self.use_positional_encoding and self.pos_embedding is not None:
             # Expand pos_embedding to full batch, then concatenate on last (embedding) dimension
            emb = torch.cat([emb, self.pos_embedding[:, :emb.size(1), :].expand(emb.size(0), -1, -1).to(emb.device)], dim=-1)
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=emb.device), diagonal=1).bool().unsqueeze(0)
        hidden = self.attn_block(emb, mask)
        residual = hidden 
        logits = self.output_proj(residual) 
        return logits


# Manual Gradient Computation Functions
def compute_manual_gradients(model, x, y, device):
    """
    Manually computes gradients for Wov and Wqk for verification.
    Assumes model structure matches the notebook: 1 head, specific embedding setup, etc.
    """
    model.eval() # We use existing weights, no dropout to worry about
    
    # Detach weights to ensure no autograd graph connection for manual calc
    Wqk = model.attn_block.Wqk.detach().squeeze(0) # [D, D]
    Wov = model.attn_block.Wov.detach().squeeze(0) # [D, V]
    head_dim = model.attn_block.head_dim
    
    grad_Wov = torch.zeros_like(Wov)
    grad_Wqk = torch.zeros_like(Wqk)
    
    # Replicate embedding logic
    emb = model.embedding(x) # [B, T, D]
    if model.use_positional_encoding and model.pos_embedding is not None:
         emb = torch.cat([emb, model.pos_embedding[:, :emb.size(1), :].expand(emb.size(0), -1, -1).to(device)], dim=-1)
    
    batch_size = x.size(0)
    
    for b in range(batch_size):
        seq = emb[b] # [T, D]
        target_idx = y[b, -1] # scalar
        
        # --- Forward ---
        # 1. Attention Scores
        scores = (seq @ Wqk @ seq.T) / (head_dim ** 0.5) # [T, T]
        
        T = seq.size(0)
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-1e9'))
        
        attn_weights = torch.softmax(scores, dim=-1) # [T, T]
        
        # 2. Output
        logits_before_ov = attn_weights @ seq # [T, D]
        final_logits = logits_before_ov @ Wov # [T, V]
        
        logits_last = final_logits[-1] # [V]
        
        # --- Backward ---
        y_hat = torch.softmax(logits_last, dim=-1)
        y_ohe = torch.zeros_like(y_hat)
        y_ohe[target_idx] = 1.0
        
        del_y = y_hat - y_ohe # [V]
        
        # 3. Wov Gradient
        # grad_Wov += input.T @ grad_output
        grad_Wov += logits_before_ov[-1].unsqueeze(1) @ del_y.unsqueeze(0)
        
        # 4. Wqk Gradient
        # Propagate back to Attention
        del_h = Wov @ del_y # [D]
        del_attn = seq @ del_h # [T] (dL/dA[-1,:])
        
        a_last = attn_weights[-1]
        jacobian = torch.diag(a_last) - torch.outer(a_last, a_last)
        del_scores = del_attn @ jacobian # [T]
        
        w_sum = seq.T @ del_scores # [D]
        grad_Wqk += (seq[-1].unsqueeze(1) @ w_sum.unsqueeze(0)) / (head_dim ** 0.5)

    return grad_Wov / batch_size, grad_Wqk / batch_size

def train_and_verify():
    # Setup
    context_len = 20
    vocab_size = 26
    embed_dim = 26
    num_heads = 1
    
    train_set, test_set = make_train_test_sets(num_train=100, num_test=1000, context_len=context_len, vocab_size=vocab_size)
    train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False)
    
    model = SimpleTransformerScratch(
        vocab_size=vocab_size,
        context_len=context_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_mlp=False,
        use_identity_embedding=True,
        use_identity_unembedding=True,
        use_positional_encoding=True,
        use_layernorm=False,
        remove_bias=True,
    )
    model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    test_accuracies = []
    
    epochs = 20
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # We need to capture gradients manually
        # To do this exactly as requested would require the derivation.
        # However, we can access the gradients populated by .backward().
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Forward
            logits = model(x) # [B, T, V]
            
            # Loss on last token only as per notebook
            logits_last = logits[:, -1, :] # [B, V]
            y_last = y[:, -1] # [B]
            
            loss = criterion(logits_last, y_last)
            
            # Zero grad
            optimizer.zero_grad()
            
            # Backward
            loss.backward()
            
            # For verification, we print the norm of the gradients generated by autograd
            # This serves as a baseline.
            # In a full implementation, we would compute `grad_manual` here and compare.
            
            grad_Wov_manual, grad_Wqk_manual = compute_manual_gradients(model, x, y, device)
            
            with torch.no_grad():
                if model.attn_block.Wqk.grad is not None:
                    grad_Wqk_autograd = model.attn_block.Wqk.grad.squeeze(0)
                    diff_wqk = (grad_Wqk_manual - grad_Wqk_autograd).abs().max()
                    # print(f"  Wqk Diff Max: {diff_wqk.item():.6f}")
                
                if model.attn_block.Wov.grad is not None:
                    grad_Wov_autograd = model.attn_block.Wov.grad.squeeze(0)
                    diff_wov = (grad_Wov_manual - grad_Wov_autograd).abs().max()
                    # print(f"  Wov Diff Max: {diff_wov.item():.6f}")
                
                if (batch_idx := len(train_losses)) % 10 == 0: # Print occasionally
                     print(f"  Batch {batch_idx}: Wqk Diff: {diff_wqk.item():.6f}, Wov Diff: {diff_wov.item():.6f}")
            
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits[:, -1, :].argmax(dim=-1)
                correct += (preds == y[:, -1]).sum().item()
                total += y.size(0)
        acc = correct / total
        test_accuracies.append(acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {acc:.4f}")
        
    print("Training complete.")

if __name__ == "__main__":
    train_and_verify()
