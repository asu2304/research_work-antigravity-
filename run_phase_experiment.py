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

# --- 1. Data Creation (Same as before) ---
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

def make_train_test_sets(num_train=600, num_test=100000, context_len=20, vocab_size=26):
    alphabet = [chr(65 + i) for i in range(vocab_size)]
    all_combs = []
    for i in range(vocab_size):
        for j in range(vocab_size):
            for k in range(vocab_size):
                for l in range(vocab_size):
                    for m in range(vocab_size):
                        all_combs.append((alphabet[i], alphabet[j], alphabet[k], alphabet[l], alphabet[m]))
                        
    # Shuffle and sample for train/test
    random.seed(42)
    random.shuffle(all_combs)
    train_combinations = all_combs[:num_train]
    # For large test set, we just take the next 100k (or loops if needed, but combinations are 26^5 > 11M)
    test_combinations  = all_combs[num_train : num_train + num_test]
    
    train_set = ABCAlternatingPatternDataset(train_combinations, context_len, vocab_size)
    test_set  = ABCAlternatingPatternDataset(test_combinations,  context_len, vocab_size)
    
    return train_set, test_set

# --- 2. Manual Gradient Calculations (De-coupled) ---

def compute_gradient_for_Wov(Wqk, Wov, train_loader, embedding_layer, device):
    """
    Computes gradient for Wov while treating Wqk as fixed.
    Accumulates gradient over the entire train_loader.
    """
    grad_Wov_accum = torch.zeros_like(Wov)
    count = 0
    head_dim = Wqk.size(0)
    
    # Pre-compute fixed parts if possible, but for simplicity loop batch-wise
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        B = x.size(0)
        
        # Embed
        emb = embedding_layer(x) # [B, T, D]
        # (Assuming no Positional Encoding for manual logic based on notebook search,
        # but previously used it. Notebook 10 uses `embedding_of_sequence` which implies simple embedding
        # or pos encoding. I will use simple embedding as per "simplifying" name, 
        # unless previous run_exp established pos enc. It had it. I'll stick to simple embedding 
        # to match "manual logic" purely unless gradients fail.)
        # RE-CHECK: Notebook uses `embedding_of_sequence`.
        # I'll stick to a simple embedding for now as per `SimpleTransformerScratch` passed logic.
        
        for b in range(B):
            seq = emb[b] # [T, D]
            target_idx = y[b, -1]
            
            # Forward (Wqk fixed)
            scores = (seq @ Wqk @ seq.T) / (head_dim ** 0.5)
            # Mask
            T_len = seq.size(0)
            mask = torch.triu(torch.ones(T_len, T_len, device=device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-1e9'))
            attn_weights = torch.softmax(scores, dim=-1)
            
            logits_before_ov = attn_weights @ seq # [T, D]
            final_logits = logits_before_ov @ Wov # [T, V]
            
            logits_last = final_logits[-1] # [V]
            
            # Backward
            y_hat = torch.softmax(logits_last, dim=-1)
            y_ohe = torch.zeros_like(y_hat)
            y_ohe[target_idx] = 1.0
            del_y = y_hat - y_ohe
            
            # grad_Wov += input.T @ grad_output
            # input to Wov is logits_before_ov[-1] (for the last token)
            grad_Wov_accum += logits_before_ov[-1].unsqueeze(1) @ del_y.unsqueeze(0)
            count += 1
            
    return grad_Wov_accum / count

def compute_gradient_for_Wqk(Wqk, Wov, train_loader, embedding_layer, device):
    """
    Computes gradient for Wqk while treating Wov as fixed.
    """
    grad_Wqk_accum = torch.zeros_like(Wqk)
    count = 0
    head_dim = Wqk.size(0)
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        B = x.size(0)
        emb = embedding_layer(x)
        
        for b in range(B):
            seq = emb[b]
            target_idx = y[b, -1]
            
            scores = (seq @ Wqk @ seq.T) / (head_dim ** 0.5)
            T_len = seq.size(0)
            mask = torch.triu(torch.ones(T_len, T_len, device=device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-1e9'))
            attn_weights = torch.softmax(scores, dim=-1)
            
            logits_before_ov = attn_weights @ seq
            final_logits = logits_before_ov @ Wov
            
            # Backward
            y_hat = torch.softmax(final_logits[-1], dim=-1)
            y_ohe = torch.zeros_like(y_hat)
            y_ohe[target_idx] = 1.0
            del_y = y_hat - y_ohe
            
            # Backprop to attention
            del_h = Wov @ del_y # [D]
            del_attn = seq @ del_h # [T]
            
            a_last = attn_weights[-1]
            jacobian = torch.diag(a_last) - torch.outer(a_last, a_last)
            del_scores = del_attn @ jacobian # [T]
            
            w_sum = seq.T @ del_scores # [D]
            # Gradient for Wqk
            grad_Wqk_accum += (seq[-1].unsqueeze(1) @ w_sum.unsqueeze(0)) / (head_dim ** 0.5)
            count += 1
            
    return grad_Wqk_accum / count

def evaluate(Wqk, Wov, test_loader, embedding_layer, device):
    head_dim = Wqk.size(0)
    correct = 0
    total = 0
    
    # We can batch verify for speed using PyTorch broadcasting if careful, 
    # but sequential is safer for matching manual logic EXACTLY.
    # Actually, let's use batch ops where possible for eval speed.
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        B = x.size(0)
        emb = embedding_layer(x) # [B, T, D]
        
        # Batch attention
        # Q = K = V = emb (since Wqk decomposes to QK^T but here we use Wqk direct on seq)
        # scores = x Wqk x.T ideally.
        # But we defined scores = (seq @ Wqk @ seq.T) per sequence.
        # Batched: [B, T, D] @ [D, D] -> [B, T, D]
        # Then @ [B, D, T] -> [B, T, T]
        
        x_wqk = torch.matmul(emb, Wqk) # [B, T, D]
        scores = torch.matmul(x_wqk, emb.transpose(-2, -1)) / (head_dim ** 0.5)
        
        T_len = x.size(1)
        mask = torch.triu(torch.ones(T_len, T_len, device=device), diagonal=1).bool().unsqueeze(0)
        scores = scores.masked_fill(mask, float('-1e9'))
        attn = torch.softmax(scores, dim=-1)
        
        # Out
        out = attn @ emb @ Wov # [B, T, D] @ [D, V] -> [B, T, V]
        
        preds = out[:, -1, :].argmax(dim=-1)
        correct += (preds == y[:, -1]).sum().item()
        total += B
        
    return correct / total

# --- 3. Main Experiment Loop ---

def run_phase_experiment():
    # Helper for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Parameters
    context_len = 20
    vocab_size = 26
    embed_dim = 26
    num_heads = 1
    
    iterations = 10
    freeze_steps = 5
    learning_rate = 0.01
    
    print("Generating data...")
    train_set, test_set = make_train_test_sets(num_train=600, num_test=100000, context_len=context_len, vocab_size=vocab_size)
    train_loader = DataLoader(train_set, batch_size=600, shuffle=False) # Full batch for manual gradient consistency
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    
    # Initialization
    # "Earlier initialized Wqk and Wov" -> Random Normal(0, 0.1) as per notebook
    Wqk = torch.empty(embed_dim, embed_dim, device=device)
    Wov = torch.empty(embed_dim, vocab_size, device=device)
    nn.init.normal_(Wqk, mean=0, std=0.1)
    nn.init.normal_(Wov, mean=0, std=0.1)
    
    # Use Identity embedding for simplicity as per "SimpleTransformer" logic
    embedding_layer = nn.Embedding(vocab_size, vocab_size)
    with torch.no_grad():
        embedding_layer.weight.copy_(torch.eye(vocab_size))
    embedding_layer.weight.requires_grad = False
    embedding_layer.to(device)
    
    # Lists to store history
    Wqk_history = [Wqk.clone()]
    Wov_history = [Wov.clone()]
    accuracies = []
    
    plot_x_labels = []
    
    print("Starting Phase-wise Training...")
    
    global_step = 0
    
    for iteration in range(iterations):
        
        # --- Phase 1: Train Wov (Wqk Fixed) ---
        print(f"--- Iteration {iteration+1} Phase 1 (Train Wov) ---")
        curr_Wqk = Wqk_history[-1]
        curr_Wov = Wov_history[-1].clone()
        
        for step in range(freeze_steps):
            grad_Wov = compute_gradient_for_Wov(curr_Wqk, curr_Wov, train_loader, embedding_layer, device)
            
            # Update
            curr_Wov = curr_Wov - learning_rate * grad_Wov
            
            # Evaluate
            acc = evaluate(curr_Wqk, curr_Wov, test_loader, embedding_layer, device)
            accuracies.append(acc)
            plot_x_labels.append(f"I{iteration}_OV{step}")
            print(f"  OV step {step}: Acc={acc:.4f}")
            
        Wov_history.append(curr_Wov)
        Wqk_history.append(curr_Wqk) # Unchanged
        
        # --- Phase 2: Train Wqk (Wov Fixed) ---
        print(f"--- Iteration {iteration+1} Phase 2 (Train Wqk) ---")
        curr_Wqk = Wqk_history[-1].clone()
        curr_Wov = Wov_history[-1]
        
        for step in range(freeze_steps):
            grad_Wqk = compute_gradient_for_Wqk(curr_Wqk, curr_Wov, train_loader, embedding_layer, device)
            
            # Update
            curr_Wqk = curr_Wqk - learning_rate * grad_Wqk
            
            # Evaluate
            acc = evaluate(curr_Wqk, curr_Wov, test_loader, embedding_layer, device)
            accuracies.append(acc)
            plot_x_labels.append(f"I{iteration}_QK{step}")
            print(f"  QK step {step}: Acc={acc:.4f}")
            
        Wqk_history.append(curr_Wqk)
        Wov_history.append(curr_Wov) # Unchanged
        
    # --- Plotting ---
    plt.figure(figsize=(15, 6))
    plt.plot(accuracies, marker='o')
    plt.title('Accuracy over Phase-wise Training (Alternating Wov/Wqk)')
    plt.xlabel('Steps')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    # plt.xticks(range(len(accuracies)), plot_x_labels, rotation=90) # Too dense for 100 points
    plt.tight_layout()
    plt.savefig('phase_training_accuracy.png')
    print("Plot saved to phase_training_accuracy.png")

if __name__ == "__main__":
    run_phase_experiment()
