#!/bin/bash

gpu=""
lora_rank=""
learning_rate=""
layers=""
extra_name=""
annotation=""
wandb=""
quant_ema_decay=0.99
eval_step_zero=0

qlora=/thayerfs/home/f004h3t/Workspaces/multi-modal-generative-ai/storage/Workspaces/hierarchical-residuals/qlora-hr-vqlora/qlora.py

usage() { echo "Usage: $0 -g <gpu_no> -r <lora rank> -l <learning rate (og: 0.0002)> [-w <use wanddb> -n <arr layer ranks> -d <quant ema decay> -e <extra name> -a <annotation> -t <eval step zero>" 1>&2; exit 1; }

## if w flag is present, set wandb to true
while getopts ":g:r:l:d:n:e:a:w:t:" o; do
    case "${o}" in
        g)
            gpu=${OPTARG}
            ;;
        r)
            lora_rank=${OPTARG}
            ;;
        l)
            learning_rate=${OPTARG}
            ;;
        n)
            layers=${OPTARG}
            ;;
        e)
            extra_name=${OPTARG}
            ;;
        a)
            annotation=${OPTARG}
            ;;
        w)
            wandb="--report_to wandb"
            ;;
        d)
            quant_ema_decay=${OPTARG}
            ;;
        t)
            eval_step_zero=1
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

echo "lora_rank = ${lora_rank}"
echo "learning_rate = ${learning_rate}"
echo "layers = ${layers}"
echo "annotation = ${annotation}"
echo "Recording to wandb? = ${wandb}"
echo "quant_ema_decay = ${quant_ema_decay}"
echo "eval_step_zero = ${eval_step_zero}"

if [ -z "${layers}" ] || [ -z "${lora_rank}" ] || [ -z "${learning_rate}" ] || [ -z "${gpu}" ]; then
    usage
fi

if [ -z "${extra_name}" ]; then
    name="r${lora_rank}_l${learning_rate}_n${layers}_d${quant_ema_decay}"
else
    name="r${lora_rank}_l${learning_rate}_n${layers}_d${quant_ema_decay}_${extra_name}"
fi

echo "name = {$name}"

CUDA_VISIBLE_DEVICES=$gpu python $qlora \
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
    --max_steps 1850 \
    --eval_steps 187 \
    --learning_rate $learning_rate \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --quant_ema_decay $quant_ema_decay \
    --eval_step_zero $eval_step_zero \
    --layers $layers \
    --seed 0 \
    $wandb \
    --run_name $name \
    --gpu $gpu \
    --annotation $annotation
