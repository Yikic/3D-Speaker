import os
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class AttentionConstraintPropagation(nn.Module):
    """
    Reverts to the classic E2CP algorithm (closed-form propagation) to retain 
    the original logical structure, but introduce an MLP-based dynamic 
    adjustment specifically on the inversion matrix (inv_mat).
    """
    def __init__(self, emb_dim=192, max_iter=20, tol=1e-5, alpha=0.8):
        super(AttentionConstraintPropagation, self).__init__()
        # 将 alpha 注册为可学习参数，并使用 inverse-sigmoid (logit) 初始化
        # 在 forward 中通过 sigmoid 将其约束在 (0, 1) 内，以保证传播矩阵的稳定性
        self.raw_alpha = nn.Parameter(torch.log(torch.tensor(alpha / (1.0 - alpha))))
        
        # We learn how to adjust inv_mat components (which correspond to graph edge diffusions) 
        # using the original embeddings' feature spaces. We map from 2*D (pair of embeddings)
        # to a single scalar multiplier value acting as a graph scaling mask.
        self.adj_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
            nn.Sigmoid() # Scale outputs between 0 and 1 (or change to Tanh/Softplus as needed)
        )

    def forward(self, embeddings, Z):
        """
        embeddings: Tensor of shape (N, D) - Speaker embeddings
        Z: Tensor of shape (N, N) - Initial predicted constraints (+1/0/-1)
        """
        N, D = embeddings.shape
        C = (Z != 0).to(embeddings.dtype)
        
        # 验证并修正 C 为正定矩阵
        # 如果有小于等于0的特征值，通过给对角线增加一个自适应的偏移量使其变成正定矩阵（代价最小的平移做法）
        eigenvalues = torch.linalg.eigvalsh(C)
        min_eig = torch.min(eigenvalues)
        if min_eig <= 0:
            epsilon = 1
            shift = -min_eig + epsilon
            # 仅在对角线上以最小的代价将其补为正定矩阵
            C = C + shift * torch.eye(N, dtype=embeddings.dtype, device=embeddings.device)
            # print(f"C is not PD. Shifted diagonal by {shift.item():.6f} to make it positive definite.")
        
        # 1. Base Similarity W: Cosine similarity normalized to [0, 1]
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        W = (torch.mm(embeddings_norm, embeddings_norm.T) + 1.0) / 2.0
        
        # 2. E2CP Step 1: Compute normalized Laplacian L_bar
        d = torch.sum(W, dim=1)
        d = torch.clamp(d, min=1e-8)
        d_inv_sqrt = torch.pow(d, -0.5)
        D_inv_sqrt = torch.diag(d_inv_sqrt)

        I = torch.eye(N, dtype=embeddings.dtype, device=embeddings.device)
        L_bar = C - I + D_inv_sqrt @ W @ D_inv_sqrt
        
        # 实时计算当前受约束的 alpha
        alpha = torch.sigmoid(self.raw_alpha)
        
        # 3. E2CP Step 2: Optimal Closed-form propagation
        inv_mat = torch.linalg.inv(C - alpha * L_bar)
        
        # 4. Adjustment of `inv_mat`: 
        # Since inv_mat is an NxN transition matrix indicating how much constraint 
        # flows from chunk i to chunk j, we predict a continuous element-wise adjustment 
        # mask M (NxN) based on the features of chunk i and chunk j.
        # We process this dynamically using OOM-safe batching in inference to prevent 
        # large explicit (N, N, emb_dim) explicit tensor caching.
        
        # Free up memory locally
        W1 = self.adj_mlp[0].weight[:, :D] # (emb_dim, D)
        W2 = self.adj_mlp[0].weight[:, D:] # (emb_dim, D)
        bias1 = self.adj_mlp[0].bias # (emb_dim)
        
        proj_i = F.linear(embeddings, W1) # (N, emb_dim)
        proj_j = F.linear(embeddings, W2) # (N, emb_dim)
        
        # Calculate final layer ahead of time
        W_last = self.adj_mlp[2].weight # (1, emb_dim)
        bias_last = self.adj_mlp[2].bias
        
        # Calculate mask memory-efficiently (vectorized loop or direct chunking)
        # Using a highly-efficient low-memory formulation:
        # Instead of `ReLU(A + B) @ W`, we can perform computations iteratively along rows if needed.
        # But we can also refactor MLP logic into something completely memory efficient:
        # We need sum_k [ ReLU(proj_i[:, k].unsqueeze(1) + proj_j[:, k].unsqueeze(0) + bias1[k]) * W_last[0, k] ]
        
        adj_mask_rows = []
        # Chunk row computations to save memory footprint
        chunk_size = 200
        for i_start in range(0, N, chunk_size):
            i_end = min(N, i_start + chunk_size)
            
            # (chunk_N, 1, emb_dim) + (1, N, emb_dim) -> (chunk_N, N, emb_dim)
            combined_chunk = proj_i[i_start:i_end].unsqueeze(1) + proj_j.unsqueeze(0) + bias1
            combined_chunk = F.relu(combined_chunk)
            
            # -> (chunk_N, N)
            mask_chunk_raw = torch.matmul(combined_chunk, W_last.T).squeeze(-1) + bias_last
            adj_mask_rows.append(mask_chunk_raw)
            
        adj_mask = torch.cat(adj_mask_rows, dim=0) # (N, N)
        adj_mask = torch.sigmoid(adj_mask) 
        
        # Apply adjustment. We can scale the global inverted transition paths
        inv_mat_adj = inv_mat * adj_mask
        
        # Formula: F* = (1-alpha)^2 * (I - alpha*L_bar)^-1 @ Z @ (I - alpha*L_bar)^-1
        F_star = ((1.0 - alpha)**2) * (inv_mat_adj @ (C*Z) @ inv_mat_adj)
        
        return F_star

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
                valid_mask = (Z_gt != 0)
                # To prevent empty graph crash:
                if valid_mask.sum() == 0:
                    continue
                    
                loss = criterion(F_star[valid_mask], Z_gt[valid_mask])
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
