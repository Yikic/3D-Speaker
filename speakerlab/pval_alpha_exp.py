import sys
import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.audio import Inference

# Adjust python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from speakerlab.utils.fileio import load_audio
from speakerlab.bin.infer_diarization import Diarization3Dspeaker, compressed_seg

def load_rttm_as_annotation(rttm_path):
    ann = Annotation()
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == 'SPEAKER':
                onset = float(parts[3])
                duration = float(parts[4])
                spk = parts[7]
                ann[Segment(onset, onset+duration)] = spk
    return ann

def find_ref_rttm(wav_id, ref_dir):
    def _normalize(name):
        stem = name
        if stem.endswith(".rttm"): stem = stem[:-5]
        if "_MS" in stem:
            prefix, suffix = stem.rsplit("_MS", 1)
            if suffix.isdigit(): stem = prefix
        return stem
    
    target = _normalize(wav_id)
    for root, dirs, files in os.walk(ref_dir):
        for f in files:
            if f.endswith('.rttm') and _normalize(f) == target:
                return os.path.join(root, f)
    return None

def build_soft_constraint_matrix(chunks, segmentations_scores, threshold):
    n = len(chunks)
    constraint_matrix = np.zeros((n, n), dtype=np.int8)
    assigned = np.zeros((n, n), dtype=np.int8)
    
    seg_window = segmentations_scores.sliding_window
    num_windows, num_frames_per_chunk, num_classes = segmentations_scores.data.shape
    frame_duration = seg_window.duration / num_frames_per_chunk
    
    for win_idx, (seg_chunk, data) in enumerate(segmentations_scores):
        win_start = seg_chunk.start
        win_end = seg_chunk.end
        
        overlapping = []
        chunk_probs = {}
        for chunk_idx, (c_st, c_ed) in enumerate(chunks):
            overlap_st = max(win_start, c_st)
            overlap_ed = min(win_end, c_ed)
            if overlap_ed > overlap_st:
                frame_st = int((overlap_st - win_start) / frame_duration)
                frame_ed = int((overlap_ed - win_start) / frame_duration)
                frame_st = max(0, min(frame_st, num_frames_per_chunk))
                frame_ed = max(0, min(frame_ed, num_frames_per_chunk))
                if frame_st < frame_ed:
                    class_activations = data[frame_st:frame_ed, :].mean(axis=0)
                    dominant_class = int(np.argmax(class_activations))
                    max_prob = class_activations[dominant_class]
                    chunk_probs[chunk_idx] = {'class': dominant_class, 'prob': max_prob}
                overlapping.append(chunk_idx)
                
        determined_chunks = [c for c in overlapping if c in chunk_probs]
        for i in range(len(determined_chunks)):
            for j in range(i + 1, len(determined_chunks)):
                ci = determined_chunks[i]
                cj = determined_chunks[j]
                
                if chunk_probs[ci]['prob'] > threshold and chunk_probs[cj]['prob'] > threshold:
                    relation = 1 if chunk_probs[ci]['class'] == chunk_probs[cj]['class'] else -1
                    if assigned[ci, cj] == 0:
                        constraint_matrix[ci, cj] = relation
                        constraint_matrix[cj, ci] = relation
                        assigned[ci, cj] = 1
                        assigned[cj, ci] = 1
                    else:
                        if constraint_matrix[ci, cj] != relation:
                            constraint_matrix[ci, cj] = 0
                            constraint_matrix[cj, ci] = 0
    return constraint_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/dataset_thres.yaml')
    parser.add_argument('--out_dir', type=str, default='results/results_pval_alpha_exp')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hf_token = os.environ.get("HuggingFaceToken", None)
    
    # Initialize pipeline
    pipeline = Diarization3Dspeaker(device=device, include_overlap=True, hf_access_token=hf_token, use_constraint=True, include_overlap_post=True)
    
    pvals = np.linspace(0.005, 0.05, 10)
    alphas = np.linspace(0.0, 1.0, 11)
    metric = DiarizationErrorRate()
    
    for ds_name, items in config_data.items():
        print(f"\n========== Processing dataset: {ds_name} ==========")
        ds_conf = {}
        for item in items:
            ds_conf.update(item)
            
        wav_dir = ds_conf['wav_dir']
        wav_list_path = ds_conf['wav_list']
        ref_rttms_dir = ds_conf['ref_rttms']
        threshold = float(ds_conf['threshold'])
        
        wavs = []
        with open(wav_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('/'):
                    wavs.append(line)
                else:
                    wavs.append(os.path.join(wav_dir, line))
                    
        confusion_matric = np.zeros((len(pvals), len(alphas)))
        total_matric = np.zeros((len(pvals), len(alphas)))
        
        for wav_path in tqdm(wavs, desc=f"Evaluating {ds_name}"):
            wav_id = os.path.basename(wav_path).rsplit('.', 1)[0]
            ref_path = find_ref_rttm(wav_id, ref_rttms_dir)
            if ref_path is None:
                print(f"[Warning] No RTTM found for {wav_id}, skipping.")
                continue
            ref_ann = load_rttm_as_annotation(ref_path)
            
            # Step 1: Load audio & Extract segmentations once
            wav_data = load_audio(wav_path, None, pipeline.fs)
            
            segmentations_hard = pipeline.segmentation_model({'waveform': wav_data, 'sample_rate': pipeline.fs}, soft=False)
            segmentations_soft = pipeline.segmentation_model({'waveform': wav_data, 'sample_rate': pipeline.fs}, soft=True)
            frame_windows = pipeline.segmentation_model.model.receptive_field
            count = Inference.aggregate(
                np.sum(segmentations_hard, axis=-1, keepdims=True),
                frame_windows, hamming=False, missing=0.0, skip_average=False
            )
            count.data = np.rint(count.data).astype(np.uint8)
            
            # Step 2: Prepare chunks and extract embeddings once
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
                
            embeddings = pipeline.do_emb_extraction(chunks, wav_data)
            constraint_matrix = build_soft_constraint_matrix(chunks, segmentations_soft, threshold)
            
            # Step 3: Grid search over pval and alpha
            for p_idx, pval in enumerate(pvals):
                for a_idx, alpha in enumerate(alphas):
                    cluster_labels = pipeline.cluster(
                        embeddings, 
                        speaker_num=None, 
                        constraint_matrix=constraint_matrix,
                        pval=float(pval),
                        alpha=float(alpha)
                    )
                    spk_num = cluster_labels.max() + 1
                    
                    output_field_labels = [[i[0], i[1], int(j)] for i, j in zip(chunks, cluster_labels)]
                    output_field_labels = compressed_seg(output_field_labels)
                    
                    # binary = pipeline.post_process(output_field_labels, spk_num, segmentations_hard, count)
                    # timestamps = [count.sliding_window[i].middle for i in range(binary.shape[0])]
                    # output_field_labels = pipeline.binary_to_segs(binary, timestamps)
                    
                    hyp_ann = Annotation()
                    for seg in output_field_labels:
                        hyp_ann[Segment(seg[0], seg[1])] = f"SPEAKER_{seg[2]}"
                        
                    components = metric.compute_components(ref_ann, hyp_ann)
                    confusion_matric[p_idx, a_idx] += components.get("confusion", 0.0)
                    total_matric[p_idx, a_idx] += components.get("total", 0.0)
                    
        conf_rate = np.divide(confusion_matric, total_matric, out=np.zeros_like(confusion_matric), where=total_matric!=0)
        
        best_idx = np.unravel_index(np.argmin(conf_rate), conf_rate.shape)
        best_p = pvals[best_idx[0]]
        best_a = alphas[best_idx[1]]
        print(f"Optimal for {ds_name}: pval={best_p:.4f}, alpha={best_a:.4f} with Speaker Confusion={conf_rate[best_idx]:.4f}")
        
        # 打印一下整体的平均约束传播耗时
        if hasattr(pipeline.cluster, 'cluster') and hasattr(pipeline.cluster.cluster, 'e2cp_times'):
            times = pipeline.cluster.cluster.e2cp_times
            if times:
                avg_time = sum(times) / len(times)
                print(f"[Profiling] Average E2CP constraint propagation time for {ds_name}: {avg_time:.4f} seconds (Total queries: {len(times)})")
            pipeline.cluster.cluster.e2cp_times = [] # clear for the next dataset
            
        # Plot 3D surface
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        P, A = np.meshgrid(pvals, alphas, indexing='ij')
        surf = ax.plot_surface(P, A, conf_rate, cmap='viridis')
        ax.set_xlabel('pval')
        ax.set_ylabel('alpha')
        ax.set_zlabel('Speaker Confusion')
        plt.title(f'Speaker Confusion Grid Search ({ds_name})')
        fig.colorbar(surf, ax=ax, label='Confusion Rate')
        
        save_path = os.path.join(args.out_dir, f'{ds_name}_grid_search_3d.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved 3D plot to {save_path}")

        # Plot 2D slice for best alpha
        plt.figure(figsize=(8, 6))
        plt.plot(pvals, conf_rate[:, best_idx[1]], marker='o')
        plt.xlabel('pval')
        plt.ylabel('Speaker Confusion')
        plt.title(f'Speaker Confusion vs pval (alpha={best_a:.4f}) on {ds_name}')
        plt.grid(True)
        slice_p_path = os.path.join(args.out_dir, f'{ds_name}_slice_best_alpha.png')
        plt.savefig(slice_p_path)
        plt.close()
        print(f"Saved 2D slice (best alpha) to {slice_p_path}")

        # Plot 2D slice for best pval
        plt.figure(figsize=(8, 6))
        plt.plot(alphas, conf_rate[best_idx[0], :], marker='s', color='orange')
        plt.xlabel('alpha')
        plt.ylabel('Speaker Confusion')
        plt.title(f'Speaker Confusion vs alpha (pval={best_p:.4f}) on {ds_name}')
        plt.grid(True)
        slice_a_path = os.path.join(args.out_dir, f'{ds_name}_slice_best_pval.png')
        plt.savefig(slice_a_path)
        plt.close()
        print(f"Saved 2D slice (best pval) to {slice_a_path}")

if __name__ == '__main__':
    main()
