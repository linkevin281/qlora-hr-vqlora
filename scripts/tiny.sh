#!/bin/bash

codebook_size=""
lora_rank=""
learning_rate=""
codebook_layers=3
extra_name=""

usage() { echo "Usage: $0 -r <lora rank> -c <codebook size> -l <learning rate (og: 0.0002)> [-n <no. codebook layers>] [-e <extra name>]" 1>&2; exit 1; }

while getopts ":c:r:l:n:e:" o; do
    case "${o}" in
        c)
            codebook_size=${OPTARG}
            ;;
        r)
            lora_rank=${OPTARG}
            ;;
        l)
            learning_rate=${OPTARG}
            ;;
        n)
            codebook_layers=${OPTARG}
            ;;
        e)
            extra_name=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

echo "codebook_size = ${codebook_size}"
echo "lora_rank = ${lora_rank}"
echo "learning_rate = ${learning_rate}"
echo "codebook_layers = ${codebook_layers}"


if [ -z "${codebook_size}" ] || [ -z "${lora_rank}" ] || [ -z "${learning_rate}" ]; then
    usage
fi

if [ -z "${extra_name}" ]; then
    name="r${lora_rank}_c${codebook_size}_l${learning_rate}_n${codebook_layers}"
else
    name="r${lora_rank}_c${codebook_size}_l${learning_rate}_n${codebook_layers}_${extra_name}"
fi

echo "name = {$name}"

CUDA_VISIBLE_DEVICES='0' python qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir /thayerfs/home/f004h3t/Workspaces/multi-modal-generative-ai/storage/real_runs/$name \
    --logging_steps 1 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 500 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
    --lora_r $lora_rank \
    --codebook_size $codebook_size \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset oasst1 \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 1875 \
    --eval_steps 187 \
    --learning_rate $learning_rate \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --report_to wandb \
    --run_name $name