#!/bin/bash

# Ensure running in speakerlab directory
cd "$(dirname "$0")"

CONFIG_FILE="config/dataset.yaml"
BASE_OUT_DIR="${1:-results/result_seg_eval}"

if [ -z "$HuggingFaceToken" ]; then
    echo "Warning: HuggingFaceToken environment variable is not set"
fi

dataset_name=""
wav_dir=""
wav_list=""
ref_rttms=""

process_dataset() {
    if [ -n "$dataset_name" ] && [ -n "$wav_dir" ] && [ -n "$wav_list" ] && [ -n "$ref_rttms" ]; then
        echo "========== Evaluating dataset: $dataset_name =========="
        out_dir="$BASE_OUT_DIR/$dataset_name"
        mkdir -p "$out_dir"
        
        tmp_wav_list="$out_dir/tmp_wav_list.txt"
        > "$tmp_wav_list"
        
        if [ -f "$wav_list" ]; then
            while IFS= read -r wav_file || [ -n "$wav_file" ]; do
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
            return
        fi
        
        # Determine RTTM inputs
        rttm_arg=""
        if [ -d "$ref_rttms" ]; then
            rttm_arg=$(find "$ref_rttms" -name '*.rttm' | paste -sd,)
        elif [ -f "$ref_rttms" ]; then
            rttm_arg="$ref_rttms"
        fi
        
        if [ -z "$rttm_arg" ]; then
             echo "Error: No reference RTTM files found in $ref_rttms"
             return
        fi
        
        # Run segmentation constraint evaluation
        cmd_eval="python bin/eval_seg_constraints.py --wav \"$tmp_wav_list\" --out_dir \"$out_dir\" --ref_rttm \"$rttm_arg\""
        
        if [ -n "$HuggingFaceToken" ]; then
            cmd_eval="$cmd_eval --hf_access_token \"$HuggingFaceToken\""
        fi
        
        echo "Running constraint evaluation: $cmd_eval"
        eval $cmd_eval
        
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
    
    if [[ ! "$trimmed" == "-"* ]]; then
        process_dataset
        dataset_name="${trimmed%:}"
    else
        key=$(echo "$trimmed" | awk -F': ' '{print $1}' | sed 's/^- //')
        val=$(echo "$trimmed" | awk -F': ' '{print $2}')
        
        case "$key" in
            wav_dir) wav_dir="$val" ;;
            wav_list) wav_list="$val" ;;
            ref_rttms) ref_rttms="$val" ;;
        esac
    fi
done < "$CONFIG_FILE"

# Process the last dataset
process_dataset
