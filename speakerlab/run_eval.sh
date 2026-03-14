#!/bin/bash

# 确保在speakerlab目录下运行
cd "$(dirname "$0")"

CONFIG_FILE="config/dataset.yaml"
BASE_OUT_DIR="${1:-results}"
USE_CONSTRAINT="${2:-false}"
USE_OVERLAP_POST="${3:-false}"

# 读取并简单解析 YAML 文件
dataset_name=""
wav_dir=""
wav_list=""
ref_rttms=""

process_dataset() {
    if [ -n "$dataset_name" ] && [ -n "$wav_dir" ] && [ -n "$wav_list" ] && [ -n "$ref_rttms" ]; then
        echo "========== Processing dataset: $dataset_name =========="
        out_dir="$BASE_OUT_DIR/$dataset_name"
        mkdir -p "$out_dir"
        
        # Construct an absolute path wave list since infer_diarization.py expects it
        tmp_wav_list="$out_dir/tmp_wav_list.txt"
        > "$tmp_wav_list" # 清空/创建临时文件
        
        if [ -f "$wav_list" ]; then
            while IFS= read -r wav_file || [ -n "$wav_file" ]; do
                # Trim whitespace
                wav_file=$(echo "$wav_file" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
                if [ -n "$wav_file" ]; then
                    if [[ "$wav_file" == /* ]]; then
                        echo "$wav_file" >> "$tmp_wav_list"
                    else
                        echo "$wav_dir/$wav_file" >> "$tmp_wav_list"
                    fi
                fi
            done < "$wav_list"
        else
            echo "Warning: wav_list $wav_list not found!"
        fi
        
        # Run diarization
        cmd_infer="python bin/infer_diarization.py --wav \"$tmp_wav_list\" --out_dir \"$out_dir\" --include_overlap --hf_access_token $HuggingFaceToken"
        if [ "$USE_CONSTRAINT" = "true" ]; then
            cmd_infer="$cmd_infer --use_constraint"
        fi
        if [ "$USE_OVERLAP_POST" = "true" ]; then
            cmd_infer="$cmd_infer --include_overlap_post"
        fi
        echo "Running inference: $cmd_infer"
        eval $cmd_infer
        
        # Run DER computation
        der_out_png="$out_dir/der_hist.png"
        der_out_txt="$out_dir/der_metrics.txt"
        cmd_der="python metrics/der.py --ref_dir \"$ref_rttms\" --hyp_dir \"$out_dir\" --out_dir \"$out_dir\" > \"$der_out_txt\""
        echo "Running DER evaluation: $cmd_der"
        eval $cmd_der
        
        echo -e "Results saved to $out_dir\n"
        
        # Reset variables for the next dataset
        dataset_name=""
        wav_dir=""
        wav_list=""
        ref_rttms=""
    fi
}

while IFS= read -r line || [ -n "$line" ]; do
    # Trim leading and trailing whitespace
    trimmed=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    
    # Skip empty lines
    [ -z "$trimmed" ] && continue
    
    # 如果不是以 '-' 开头，说明是一个新的数据集名字 (例如 "aishell4:")
    if [[ ! "$trimmed" == "-"* ]]; then
        process_dataset
        dataset_name="${trimmed%:}"
    else
        # 解析如 "- wav_dir: /path"
        key=$(echo "$trimmed" | awk -F': ' '{print $1}' | sed 's/^- //')
        val=$(echo "$trimmed" | awk -F': ' '{print $2}')
        
        case "$key" in
            wav_dir) wav_dir="$val" ;;
            wav_list) wav_list="$val" ;;
            ref_rttms) ref_rttms="$val" ;;
        esac
    fi
done < "$CONFIG_FILE"

# 处理最后一个数据集
process_dataset

