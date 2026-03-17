import os
import subprocess
import re
from pathlib import Path
import matplotlib.pyplot as plt

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        lines = f.readlines()
    
    cfg = {}
    for line in lines:
        line = line.strip()
        if line.startswith('- wav_dir:'):
            cfg['wav_dir'] = line.split(':', 1)[1].strip()
        elif line.startswith('- wav_list:'):
            cfg['wav_list'] = line.split(':', 1)[1].strip()
        elif line.startswith('- ref_rttms:'):
            cfg['ref_rttms'] = line.split(':', 1)[1].strip()
    return cfg

def main():
    # Ensure we are in speakerlab dir
    base_dir = Path(__file__).parent.absolute()
    os.chdir(base_dir)

    config_file = "config/ami.yaml"
    cfg = load_config(config_file)
    
    wav_dir = cfg['wav_dir']
    wav_list = cfg['wav_list']
    ref_rttms = cfg['ref_rttms']

    # Find ref rttms files for constraint
    ref_rttm_files = ",".join([str(p) for p in Path(ref_rttms).glob('*.rttm')])
    
    # Prepare tmp wav list (absolute paths)
    base_out_dir = Path("results/gt_exp")
    base_out_dir.mkdir(parents=True, exist_ok=True)
    tmp_wav_list = base_out_dir / "tmp_wav_list.txt"
    
    with open(wav_list, 'r') as f_in, open(tmp_wav_list, 'w') as f_out:
        for line in f_in:
            wav_file = line.strip()
            if not wav_file:
                continue
            if wav_file.startswith('/'):
                f_out.write(wav_file + '\n')
            else:
                f_out.write(f"{wav_dir}/{wav_file}\n")

    alphas = [0.0, 0.2, 0.5, 0.8, 1.0]
    # constraint_ratios: 0.0 to 1.0 with 0.1 step
    constraint_ratios = [round(i * 0.1, 1) for i in range(11)]

    results = {a: [] for a in alphas}

    hf_token = os.environ.get("HuggingFaceToken", "")

    for alpha in alphas:
        for cr in constraint_ratios:
            print(f"========== Processing alpha: {alpha}, constraint_ratio: {cr} ==========")
            out_dir = base_out_dir / f"alpha_{alpha}_cr_{cr}"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Run inference
            infer_cmd = [
                "python", "bin/infer_diarization.py",
                "--wav", str(tmp_wav_list),
                "--out_dir", str(out_dir),
                "--include_overlap",
                "--use_constraint",
                "--ref_rttm", ref_rttm_files,
                "--alpha", str(alpha),
                "--constraint_ratio", str(cr)
            ]
            if hf_token:
                infer_cmd.extend(["--hf_access_token", hf_token])
            
            print("Running inference:", " ".join(infer_cmd))
            subprocess.run(infer_cmd, check=True)
            
            # 2. Run DER evaluation
            der_cmd = [
                "python", "metrics/der.py",
                "--ref_dir", str(ref_rttms),
                "--hyp_dir", str(out_dir),
                "--out_dir", str(out_dir)
            ]
            print("Running DER evaluation:", " ".join(der_cmd))
            
            # Capture output to parse confusion
            run_res = subprocess.run(der_cmd, check=True, capture_output=True, text=True)
            print(run_res.stdout)
            
            # Parse output for Weighted average confusion
            match = re.search(r"Weighted average confusion:\s*([0-9.]+)", run_res.stdout)
            if match:
                confusion = float(match.group(1)) * 100
            else:
                print(f"Warning: Could not parse confusion for alpha={alpha}, cr={cr}. Setting to 0.0.")
                confusion = 0.0
            
            results[alpha].append(confusion)

    # 3. Plot results
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['b', 'g', 'r', 'c', 'm']
    
    for idx, alpha in enumerate(alphas):
        plt.plot(constraint_ratios, results[alpha], 
                 marker=markers[idx], color=colors[idx], 
                 label=f'alpha={alpha}')
        
    plt.xlabel('Constraint Ratio')
    plt.ylabel('Speaker Confusion(%)')
    plt.title('Speaker Confusion vs Constraint Ratio for varying Alpha')
    plt.legend(title='Alpha Values')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(constraint_ratios)
    
    plot_path = base_out_dir / "confusion_plot.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == '__main__':
    main()
