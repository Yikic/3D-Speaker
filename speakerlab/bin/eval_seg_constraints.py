import os
import sys
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from scipy import optimize
import json
import matplotlib.pyplot as plt

import torch
import torch.multiprocessing as mp

sys.path.append('%s/../..'%os.path.dirname(os.path.abspath(__file__)))
from speakerlab.utils.utils import silent_print
from speakerlab.utils.fileio import load_audio

os.environ['MODELSCOPE_LOG_LEVEL'] = '40'
warnings.filterwarnings("ignore")

from pyannote.audio import Inference, Model
from diarizen.pipelines.inference import DiariZenPipeline

parser = argparse.ArgumentParser(description='Evaluate segmentation constraints.')
parser.add_argument('--wav', type=str, required=True, help='Input wavs')
parser.add_argument('--out_dir', type=str, required=True, help='Out results dir')
parser.add_argument('--hf_access_token', type=str, default=None, help='hf_access_token')
parser.add_argument('--ref_rttm', type=str, required=True, help='Reference RTTM file(s), separated by comma')
parser.add_argument('--nprocs', default=1, type=int, help='Num of procs')

def get_segmentation_model(use_auth_token, device=None):
    segmentation_params = {
        'segmentation':'pyannote/segmentation-3.0',
        'segmentation_batch_size':32,
        'use_auth_token':use_auth_token,
        }
    model = Model.from_pretrained(
        segmentation_params['segmentation'], 
        use_auth_token=segmentation_params['use_auth_token'], 
        strict=False,
        )
    segmentation = Inference(
        model,
        duration=model.specifications.duration,
        step=0.1 * model.specifications.duration,
        skip_aggregation=True,
        batch_size=segmentation_params['segmentation_batch_size'],
        device=device,
        )
    return segmentation

# Load RTTM utility
def load_rttm(rttm_path):
    rttm_data = {}
    for r_path in rttm_path.split(','):
        r_path = r_path.strip()
        if not r_path: continue
        with open(r_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:
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
    for seg in ref_segs:
        overlap = max(0, min(c_ed, seg['end']) - max(c_st, seg['start']))
        if overlap > max_overlap:
            max_overlap = overlap
            best_spk = seg['speaker']
    return best_spk

def get_valid_field(count):
    valid_field = []
    start = None
    for i, (c, data) in enumerate(count):
        if data.item()==0 or i==len(count)-1:
            if start is not None:
                end = c.middle
                valid_field.append([start, end])
                start = None
        else:
            if start is None:
                start = c.middle
    return valid_field

def chunk_segments(st, ed, dur=1.5, step=0.75):
    chunks = []
    subseg_st = st
    while subseg_st + dur < ed + step:
        subseg_ed = min(subseg_st + dur, ed)
        chunks.append([subseg_st, subseg_ed])
        subseg_st += step
    return chunks

def evaluate_wav(wav, rttm_data, segmentation_model, thresholds, device):
    fs = 16000
    wav_data = load_audio(wav, None, fs)
    wav_id = os.path.basename(wav).rsplit('.', 1)[0]
    ref_segs = rttm_data.get(wav_id, [])
    
    # segmentation_scores
    segmentations_scores = segmentation_model({'waveform':wav_data, 'sample_rate': fs}, soft=True)
    frame_windows = segmentation_model.model.receptive_field
    
    # get valid fields (VSD)
    segmentations_hard = segmentation_model({'waveform':wav_data, 'sample_rate': fs}, soft=False)
    count = Inference.aggregate(np.sum(segmentations_hard, axis=-1, keepdims=True), frame_windows, hamming=False, missing=0.0, skip_average=False)
    count.data = np.rint(count.data).astype(np.uint8)
    
    vad_time = get_valid_field(count)
    chunks = [c for (st, ed) in vad_time for c in chunk_segments(st, ed)]
    
    n = len(chunks)
    if n < 2:
        return {th: {'TP_pos': 0, 'FP_pos': 0, 'FN_pos': 0, 'TN_pos': 0, 'Support': 0, 'Total_GT_pos': 0} for th in thresholds}
        
    chunk_ref_speakers = {}
    for i, chunk in enumerate(chunks):
        spk = get_chunk_ref_speaker(chunk, ref_segs)
        if spk is not None:
            chunk_ref_speakers[i] = spk

    seg_window = segmentations_scores.sliding_window
    num_windows, num_frames_per_chunk, num_classes = segmentations_scores.data.shape
    frame_duration = seg_window.duration / num_frames_per_chunk

    results = {th: {'TP_pos': 0, 'FP_pos': 0, 'FN_pos': 0, 'TN_pos': 0, 'Support': 0, 'Total_GT_pos': 0} for th in thresholds}
    
    assigned_already = set()

    for win_idx, (seg_chunk, data) in enumerate(segmentations_scores):
        win_start = seg_chunk.start
        win_end = seg_chunk.end
        
        overlapping = []
        chunk_probs = {}
        for chunk_idx, (c_st, c_ed) in enumerate(chunks):
            overlap_st = max(win_start, c_st)
            overlap_ed = min(win_end, c_ed)
            if overlap_ed > overlap_st:
                overlapping.append(chunk_idx)
                
                frame_st = int((overlap_st - win_start) / frame_duration)
                frame_ed = int((overlap_ed - win_start) / frame_duration)
                frame_st = max(0, min(frame_st, num_frames_per_chunk))
                frame_ed = max(0, min(frame_ed, num_frames_per_chunk))
                if frame_st < frame_ed:
                    # avg probability over overlapping frames
                    class_activations = data[frame_st:frame_ed, :].mean(axis=0)
                    dominant_class = int(np.argmax(class_activations))
                    max_prob = class_activations[dominant_class]
                    chunk_probs[chunk_idx] = {'class': dominant_class, 'prob': max_prob}

        determined_chunks = [c for c in overlapping if c in chunk_probs and c in chunk_ref_speakers]
        
        for i in range(len(determined_chunks)):
            for j in range(i + 1, len(determined_chunks)):
                ci, cj = determined_chunks[i], determined_chunks[j]
                if (ci, cj) in assigned_already or (cj, ci) in assigned_already:
                    continue
                assigned_already.add((ci, cj))

                gt_same = (chunk_ref_speakers[ci] == chunk_ref_speakers[cj])

                for th in thresholds:
                    if chunk_probs[ci]['prob'] > th and chunk_probs[cj]['prob'] > th:
                        pred_same = (chunk_probs[ci]['class'] == chunk_probs[cj]['class'])
                        results[th]['Support'] += 1 # A valid constraint was generated
                        
                        if pred_same:
                            if gt_same:
                                results[th]['TP_pos'] += 1
                            else:
                                results[th]['FP_pos'] += 1
                        else:
                            if not gt_same:
                                results[th]['TN_pos'] += 1
                            else:
                                results[th]['FN_pos'] += 1
                    
                    if gt_same:
                        results[th]['Total_GT_pos'] += 1

    return results

def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    segmentation_model = get_segmentation_model(args.hf_access_token, device)
    rttm_data = load_rttm(args.ref_rttm)
    
    with open(args.wav, 'r') as f:
        wav_list = [line.strip() for line in f if line.strip()]
        
    thresholds = np.linspace(0.0, 1.0, 210)
    
    total_results = {th: {'TP_pos': 0, 'FP_pos': 0, 'FN_pos': 0, 'TN_pos': 0, 'Support': 0, 'Total_GT_pos': 0} for th in thresholds}
    
    for wav_path in tqdm(wav_list, desc="Evaluating"):
        res = evaluate_wav(wav_path, rttm_data, segmentation_model, thresholds, device)
        for th in thresholds:
            for k in total_results[th]:
                total_results[th][k] += res[th][k]
    
    precisions = []
    recalls = []
    supports = []
    
    accuracies = []
    
    os.makedirs(args.out_dir, exist_ok=True)
    res_txt_path = os.path.join(args.out_dir, 'metrics_results.txt')
    f_res = open(res_txt_path, 'w')
    f_res.write("Threshold\tPrecision(ML)\tRecall(ML)\tAccuracy\tSupport\n")
    
    for th in thresholds:
        tp = total_results[th]['TP_pos']
        fp = total_results[th]['FP_pos']
        tn = total_results[th]['TN_pos']
        fn = total_results[th]['FN_pos']
        total_gt = total_results[th]['Total_GT_pos']
        support = total_results[th]['Support']
        
        # PR Curve: Precision of Must-Link
        precision_ml = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_ml = tp / total_gt if total_gt > 0 else 0
        
        # Accuracy of all constraints (+1 and -1)
        accuracy = (tp + tn) / support if support > 0 else 0
        
        # Hardcode precision to 1 when threshold is 1 (or very close to 1)
        if th >= 0.999:
            accuracy = 1.0
            precision_ml = 1.0
        
        precisions.append(precision_ml)
        recalls.append(recall_ml)
        accuracies.append(accuracy)
        supports.append(support)
        
        log_str = f"Threshold: {th:.2f} | Precision (ML): {precision_ml:.4f}, Recall (ML): {recall_ml:.4f}, Accuracy: {accuracy:.4f}, Support: {support}"
        print(log_str)
        f_res.write(f"{th:.2f}\t{precision_ml:.4f}\t{recall_ml:.4f}\t{accuracy:.4f}\t{support}\n")
    
    f_res.close()
    
    # Plot P-R curve
    plt.figure()
    plt.plot(recalls, precisions, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve of Must-Link Constraints')
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, 'pr_curve.png'))
    
    # Plot Threshold vs Accuracy/Support
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('softmax threshold')
    ax1.set_ylabel('accuracy', color='tab:blue')
    plot1 = ax1.plot(thresholds, accuracies, linestyle='None', marker='o', markerfacecolor='none', color='tab:blue', label='accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('support', color='tab:orange')
    plot2 = ax2.plot(thresholds, supports, linestyle='None', marker='s', markerfacecolor='none', color='tab:orange', label='support')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    # Add legend
    plots = plot1 + plot2
    labels = [l.get_label() for l in plots]
    ax1.legend(plots, labels, loc='center left')
    
    fig.tight_layout()
    # plt.title('Threshold vs Precision and Support Set Size')
    plt.savefig(os.path.join(args.out_dir, 'threshold_vs_acc_supp.png'))

if __name__ == '__main__':
    main()
