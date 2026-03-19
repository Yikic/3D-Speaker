import os
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class AttentionConstraintPropagation(nn.Module):
    def __init__(self, emb_dim, max_iter=20, tol=1e-5):
        super(AttentionConstraintPropagation, self).__init__()
        # W_a in the formula: weights for concatenated feature vectors
        self.Wa = nn.Linear(emb_dim * 2, 1, bias=False)
        # Learnable spatial perception time decay hyperparameters
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.max_iter = max_iter
        self.tol = tol

    def forward(self, embeddings, Z):
        """
        embeddings: Tensor of shape (N, D) - Speaker embeddings
        Z: Tensor of shape (N, N) - Initial predicted constraints (+1/0/-1)
        """
        N, D = embeddings.shape
        
        # Generate normalized segment index t evenly spaced in [0, 1]
        t = torch.linspace(0, 1, steps=N, dtype=embeddings.dtype, device=embeddings.device).unsqueeze(1)
        
        # Original: time_penalty = beta * exp(-gamma * |t_i - t_j|) (closer = larger weight)
        # New requirement: time_penalty = beta * (1 - exp(-gamma * |t_i - t_j|)) (closer = smaller weight)
        # The smaller the distance |t_i - t_j|, the larger the exp(...) term, and the smaller (1 - exp(...)) becomes.
        # This penalizes temporally adjacent segments, allocating them less attention base weight.
        t_dist = torch.abs(t - t.T)
        time_penalty = self.beta * (1.0 - torch.exp(-self.gamma * t_dist))
        
        # Optimize memory usage by broadcasting during linear projection 
        # instead of explicitly creating large (N, N, 2*D) concat tensors.
        # Let W_a = [W_a1, W_a2]
        # W_a(e_i || e_j) = W_a1(e_i) + W_a2(e_j)
        # where W_a1 and W_a2 are the first and second halves of W_a weights
        W_a1 = self.Wa.weight[:, :D]
        W_a2 = self.Wa.weight[:, D:]
        
        score_i = F.linear(embeddings, W_a1)  # (N, 1)
        score_j = F.linear(embeddings, W_a2)  # (N, 1)
        
        attn_base = score_i + score_j.T  # (N, N) via broadcasting
        
        # c_ij = LeakyReLU(Wa[e_i || e_j] + beta * exp(-gamma |t_i - t_j|))
        c = F.leaky_relu(attn_base + time_penalty)
        
        # Softmax normalization for attention weights A_ij within neighborhoods
        # Here we apply it over the sequence (axis=-1)
        A = F.softmax(c, dim=-1)
        
        # Iterative Solver for Attention Constraint Propagation
        # F^{(t+1)} = A @ F^{(t)} @ A^T + (I - A) @ Z @ (I - A)^T
        F_mat = Z.clone()
        I = torch.eye(N, dtype=embeddings.dtype, device=embeddings.device)
        
        I_A = I - A
        constant_term = I_A @ Z @ I_A.transpose(0, 1)
        
        for _ in range(self.max_iter):
            F_next = A @ F_mat @ A.transpose(0, 1) + constant_term
            # Check for convergence
            if torch.norm(F_next - F_mat, p='fro') < self.tol:
                F_mat = F_next
                break
            F_mat = F_next
            
        return F_mat

class ConstraintDataset(Dataset):
    def __init__(self, data_list):
        """
        data_list: List of dicts, each containing:
            'embeddings': (N, D) tensor
            'Z_init': (N, N) tensor of initial predicted constraints
            'Z_gt': (N, N) tensor of ground truth constraints
        """
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        return self.data_list[idx]

def collate_fn(batch):
    # Dynamic sequence lengths (N) per item, so we pass a list directly
    return batch

def train(model, dataloader, optimizer, criterion, device, epochs):
    model.train()
    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            batch_loss = 0.0
            
            for item in batch:
                emb = item['embeddings'].to(device)
                Z_init = item['Z_init'].float().to(device)
                Z_gt = item['Z_gt'].float().to(device)
                
                # Get optimized constraints F*
                F_star = model(emb, Z_init)
                
                # Apply MSE Loss. We only calculate loss where Z_gt is available (e.g. != 0)
                # You can customize valid_mask depending on whether predicting 0s is part of training.
                valid_mask = (Z_gt != 0).float()
                # To prevent empty graph crash:
                if valid_mask.sum() == 0:
                    continue
                    
                loss = criterion(F_star * valid_mask, Z_gt * valid_mask)
                batch_loss += loss
            
            if isinstance(batch_loss, float):
                continue
                
            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            
        epoch_avg_loss = total_loss/len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_avg_loss:.4f}")
        epoch_losses.append(epoch_avg_loss)
        
    return epoch_losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to precomputed .pth dataset")
    parser.add_argument('--emb_dim', type=int, default=192, help="Embedding dimension (e.g., CAMPPlus -> 192)")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_path', type=str, default='speakerlab/ckpt/attention_e2cp.pth')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading data from {args.dataset_path}...")
    raw_data = torch.load(args.dataset_path)
    dataset = ConstraintDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    model = AttentionConstraintPropagation(emb_dim=args.emb_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    print("Starting attention constraint propagation training...")
    epoch_losses = train(model, dataloader, optimizer, criterion, device, epochs=args.epochs)
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Training completed. Model saved to {args.save_path}")
    
    # Plotting the loss curve
    loss_fig_path = os.path.join(os.path.dirname(args.save_path), 'loss_curve.png')
    plt.figure()
    plt.plot(range(1, args.epochs + 1), epoch_losses, marker='o')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig(loss_fig_path)
    plt.close()
    print(f"Loss curve saved to {loss_fig_path}")

if __name__ == '__main__':
    main()
