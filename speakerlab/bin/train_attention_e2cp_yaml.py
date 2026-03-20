import os
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pyannote.audio import Inference
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

# Add speakerlab to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from speakerlab.utils.fileio import load_audio
from speakerlab.bin.infer_diarization import Diarization3Dspeaker, compressed_seg
from speakerlab.bin.train_attention_e2cp import AttentionConstraintPropagation, ConstraintDataset, collate_fn

def load_rttm_data(rttm_dir):
    rttm_data = {}
    for root, dirs, files in os.walk(rttm_dir):
        for f in files:
            if f.endswith('.rttm'):
                with open(os.path.join(root, f), 'r') as file:
                    for line in file:
                        parts = line.strip().split()
                        if len(parts) >= 8 and parts[0] == 'SPEAKER':
                            wav_id = parts[1]
                            st = float(parts[3])
                            ed = st + float(parts[4])
                            spk = parts[7]
                            if wav_id not in rttm_data: rttm_data[wav_id] = []
                            rttm_data[wav_id].append({'start': st, 'end': ed, 'speaker': spk})
    return rttm_data

def get_chunk_ref_speaker(chunk, ref_segs):
    c_st, c_ed = chunk
    max_overlap = 0
    best_spk = None
    if not ref_segs:
        return None
    for seg in ref_segs:
        overlap = max(0, min(c_ed, seg['end']) - max(c_st, seg['start']))
        if overlap > max_overlap:
            max_overlap = overlap
            best_spk = seg['speaker']
    return best_spk if max_overlap > 0 else None

def build_Z_gt(chunks, ref_segs):
    n = len(chunks)
    Z_gt = np.zeros((n, n), dtype=np.float32)
    chunk_spks = []
    for c in chunks:
        chunk_spks.append(get_chunk_ref_speaker(c, ref_segs))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                Z_gt[i, j] = 1.0
            else:
                spk_i = chunk_spks[i]
                spk_j = chunk_spks[j]
                if spk_i is not None and spk_j is not None:
                    if spk_i == spk_j:
                        Z_gt[i, j] = 1.0
                    else:
                        Z_gt[i, j] = -1.0
    return Z_gt

def build_soft_constraint_matrix(chunks, segmentations_scores, threshold):
    n = len(chunks)
    constraint_matrix = np.zeros((n, n), dtype=np.float32)
    
    seg_window = segmentations_scores.sliding_window
    num_windows, num_frames_per_chunk, num_classes = segmentations_scores.data.shape
    frame_duration = seg_window.duration / num_frames_per_chunk
    
    for win_idx, (seg_chunk, data) in enumerate(segmentations_scores):
        win_start = seg_chunk.start
        win_end = seg_chunk.end
        
        overlapping = []
        for i, (c_st, c_ed) in enumerate(chunks):
            if win_start <= c_st and win_end >= c_ed:
                # Approximate frames inside sliding window
                f_st = int((c_st - win_start) / frame_duration)
                f_ed = int((c_ed - win_start) / frame_duration)
                part = data[f_st:f_ed]
                if part.shape[0] > 0:
                    prob = np.max(np.mean(part, axis=0))
                    spk_idx = np.argmax(np.mean(part, axis=0))
                    overlapping.append({'chunk_idx': i, 'prob': prob, 'speaker_idx': spk_idx})
                    
        for i in range(len(overlapping)):
            for j in range(i+1, len(overlapping)):
                c_i = overlapping[i]
                c_j = overlapping[j]
                
                if c_i['prob'] > threshold and c_j['prob'] > threshold:
                    p_i = c_i['chunk_idx']
                    p_j = c_j['chunk_idx']
                    
                    if c_i['speaker_idx'] == c_j['speaker_idx']:
                        constraint_matrix[p_i, p_j] = 1.0
                        constraint_matrix[p_j, p_i] = 1.0
                    else:
                        constraint_matrix[p_i, p_j] = -1.0
                        constraint_matrix[p_j, p_i] = -1.0
    return constraint_matrix

def prepare_data_from_yaml(config_path, split_name, pipeline, cache_dir='speakerlab/ckpt/data_cache'):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{split_name}_data_v2.pth")
    if os.path.exists(cache_path):
        return torch.load(cache_path)
        
    with open(config_path, 'r') as f:
        conf = yaml.safe_load(f)
        
    ds_name = list(conf.keys())[0]
    if split_name not in conf[ds_name]:
        return []
        
    split_conf_list = conf[ds_name][split_name]
    split_conf = {}
    for item in split_conf_list:
        split_conf.update(item)
        
    wav_dir = split_conf['wav_dir']
    wav_list_path = split_conf['wav_list']
    ref_rttms_dir = split_conf['ref_rttms']
    threshold = float(split_conf['threshold'])
    
    wavs = []
    with open(wav_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('/'): wavs.append(line)
            else: wavs.append(os.path.join(wav_dir, line))
            
    rttm_data = load_rttm_data(ref_rttms_dir)
    prepared_list = []
    
    for wav_path in tqdm(wavs, desc=f"Preparing {split_name} data"):
        wav_id = os.path.basename(wav_path).rsplit('.', 1)[0]
        if _normalize_wav_id(wav_id) not in rttm_data:
            # Handle aishell4 naming MS/L mismatch manually or pass directly
            pass
            
        real_wav_id = None
        for k in rttm_data.keys():
            if _normalize_wav_id(k) == _normalize_wav_id(wav_id):
                real_wav_id = k
                break
                
        ref_segs = rttm_data.get(real_wav_id, [])
        if not ref_segs:
            continue
            
        wav_data = load_audio(wav_path, None, pipeline.fs)
        
        segmentations_hard = pipeline.segmentation_model({'waveform': wav_data, 'sample_rate': pipeline.fs}, soft=False)
        segmentations_soft = pipeline.segmentation_model({'waveform': wav_data, 'sample_rate': pipeline.fs}, soft=True)
        frame_windows = pipeline.segmentation_model.model.receptive_field
        
        count = Inference.aggregate(
            np.sum(segmentations_hard, axis=-1, keepdims=True),
            frame_windows, hamming=False, missing=0.0, skip_average=False
        )
        count.data = np.rint(count.data).astype(np.uint8)
        
        vad_time = []
        start = None
        for i, (c, data) in enumerate(count):
            if data.item()==0 or i==len(count)-1:
                if start is not None:
                    vad_time.append([start, c.middle])
                    start = None
            else:
                if start is None:
                    start = c.middle
        
        chunks = [c for (st, ed) in vad_time for c in pipeline.chunk(st, ed)]
        if len(chunks) < 2:
            continue
            
        embs = pipeline.do_emb_extraction(chunks, wav_data) # [N, D]
        Z_init = build_soft_constraint_matrix(chunks, segmentations_soft, threshold)
        Z_gt = build_Z_gt(chunks, ref_segs)
        
        prepared_list.append({
            'embeddings': torch.tensor(embs, dtype=torch.float32),
            'Z_init': torch.tensor(Z_init, dtype=torch.float32),
            'Z_gt': torch.tensor(Z_gt, dtype=torch.float32),
            'chunks': chunks,
            'ref_segs': ref_segs
        })
        
    torch.save(prepared_list, cache_path)
    return prepared_list

def _normalize_wav_id(name):
    stem = name
    if stem.endswith(".rttm"): stem = stem[:-5]
    if "_MS" in stem:
        prefix, suffix = stem.rsplit("_MS", 1)
        if suffix.isdigit(): stem = prefix
    if "_L" in stem:
        prefix, suffix = stem.rsplit("_L", 1)
        if suffix.isdigit(): stem = prefix
    return stem

def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            for item in batch:
                emb = item['embeddings'].to(device)
                Z_init = item['Z_init'].float().to(device)
                Z_gt = item['Z_gt'].float().to(device)
                
                F_star = model(emb, Z_init)
                valid_mask = (Z_gt != 0)
                if valid_mask.sum() == 0:
                    continue
                    
                # 仅对存在先验约束 (Z_gt != 0) 的元素计算平均 MSE
                # 如果把全图的0带入计算，梯度会被巨量的0严重稀释，导致网络参数无法更新
                loss = criterion(F_star[valid_mask], Z_gt[valid_mask])
                total_loss += loss.item()
                count += 1
    return total_loss / max(count, 1)

def eval_accuracy(model, dataloader, device):
    model.eval()
    tp_tn = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            for item in batch:
                emb = item['embeddings'].to(device)
                Z_init = item['Z_init'].float().to(device)
                Z_gt = item['Z_gt'].float().to(device)
                
                F_star = model(emb, Z_init)
                valid_mask = (Z_gt != 0)
                
                if valid_mask.sum() == 0: continue
                # Predictions
                F_pred = torch.sign(F_star)
                F_pred[F_pred == 0] = 1 # Tie break
                
                correct = ((F_pred == Z_gt) & valid_mask).sum().item()
                tp_tn += correct
                total += valid_mask.sum().item()
    
    return tp_tn / max(total, 1)

def evaluate_confusion(model, dataloader, pipeline, device):
    model.eval()
    metric = DiarizationErrorRate()
    total_confusion = 0.0
    total_total = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            for item in batch:
                if 'chunks' not in item or 'ref_segs' not in item:
                    continue
                    
                emb = item['embeddings'].to(device)
                Z_init = item['Z_init'].float().to(device)
                
                F_star = model(emb, Z_init)
                F_star_np = F_star.cpu().numpy()
                emb_np = emb.cpu().numpy()
                
                # alpha=0.0 means the static E2CP isn't executed inside cluster routine; 
                # F_star_np directly replaces the constraint matrix and updates affinity
                cluster_labels = pipeline.cluster(
                    emb_np,
                    speaker_num=None,
                    constraint_matrix=F_star_np,
                    alpha=0.7
                )
                
                chunks = item['chunks']
                ref_segs = item['ref_segs']
                
                output_field_labels = [[i[0], i[1], int(j)] for i, j in zip(chunks, cluster_labels)]
                output_field_labels = compressed_seg(output_field_labels)
                hyp_ann = Annotation()
                for seg in output_field_labels:
                    hyp_ann[Segment(seg[0], seg[1])] = f"SPEAKER_{seg[2]}"
                    
                ref_ann = Annotation()
                for seg in ref_segs:
                    ref_ann[Segment(seg['start'], seg['end'])] = seg['speaker']
                    
                components = metric.compute_components(ref_ann, hyp_ann)
                total_confusion += components.get("confusion", 0.0)
                total_total += components.get("total", 0.0)
                
    return total_confusion / max(total_total, 1e-6)

def train_and_eval(model, train_loader, eval_loader, optimizer, criterion, device, epochs, save_path, pipeline):
    train_losses = []
    eval_losses = []
    eval_confusions = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        count = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = 0.0
            valid_items = 0
            
            for item in batch:
                emb = item['embeddings'].to(device)
                Z_init = item['Z_init'].float().to(device)
                Z_gt = item['Z_gt'].float().to(device)
                
                F_star = model(emb, Z_init)
                valid_mask = (Z_gt != 0)
                if valid_mask.sum() == 0:
                    continue
                    
                # 计算损失，必须只在 valid_mask 为 True 的元素上计算
                loss = criterion(F_star[valid_mask], Z_gt[valid_mask])
                batch_loss += loss
                valid_items += 1
            
            if valid_items > 0:
                batch_loss = batch_loss / valid_items
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
                count += 1
                
        train_loss = total_loss / max(count, 1)
        eval_loss = evaluate_loss(model, eval_loader, criterion, device)
        eval_conf = evaluate_confusion(model, eval_loader, pipeline, device)
        
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        eval_confusions.append(eval_conf)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Eval Loss: {eval_loss:.4f} - Eval Confusion: {eval_conf:.4f}")
        
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Plotting Loss
    loss_fig_path = os.path.join(os.path.dirname(save_path), 'loss_curve_yaml.png')
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Train Loss')
    plt.plot(range(1, epochs + 1), eval_losses, marker='s', label='Eval Loss')
    plt.title('Training and Evaluation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_fig_path)
    plt.close()
    print(f"Loss figure saved to {loss_fig_path}")

    # Plotting Confusion
    conf_fig_path = os.path.join(os.path.dirname(save_path), 'eval_confusion_curve.png')
    plt.figure()
    plt.plot(range(1, epochs + 1), eval_confusions, marker='^', color='green', label='Eval Speaker Confusion')
    plt.title('Evaluation Speaker Confusion Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Speaker Confusion')
    plt.legend()
    plt.grid(True)
    plt.savefig(conf_fig_path)
    plt.close()
    print(f"Confusion figure saved to {conf_fig_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='speakerlab/config/aishell4.yaml')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--save_path', type=str, default='speakerlab/ckpt/attention_e2cp_yaml.pth')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hf_token = os.environ.get("HuggingFaceToken", None)
    
    pipeline = Diarization3Dspeaker(device=device, include_overlap=True, hf_access_token=hf_token)
    # pipeline = None
    print("Preparing Datasets...")
    train_data = prepare_data_from_yaml(args.yaml, 'train', pipeline)
    eval_data = prepare_data_from_yaml(args.yaml, 'eval', pipeline)
    test_data = prepare_data_from_yaml(args.yaml, 'test', pipeline)
    
    train_loader = DataLoader(ConstraintDataset(train_data), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(ConstraintDataset(eval_data), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(ConstraintDataset(test_data), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    emb_dim = train_data[0]['embeddings'].shape[1] if len(train_data) > 0 else 192
    model = AttentionConstraintPropagation(emb_dim=emb_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    print("Starting Training...")
    train_and_eval(model, train_loader, eval_loader, optimizer, criterion, device, args.epochs, args.save_path, pipeline)
    
    print("\nStarting Test Set Evaluation...")
    test_loss = evaluate_loss(model, test_loader, criterion, device)
    test_acc = eval_accuracy(model, test_loader, device)
    test_conf = evaluate_confusion(model, test_loader, pipeline, device)
    print(f"====================================")
    print(f"Test Set MSE Loss: {test_loss:.4f}")
    print(f"Test Set Constraint Accuracy: {test_acc*100:.2f}%")
    print(f"Test Set Speaker Confusion: {test_conf*100:.2f}%")
    print(f"====================================")

if __name__ == '__main__':
    main()
