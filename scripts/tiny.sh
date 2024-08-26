#!/bin/bash
## Regarding LoRA parameters, we set r = 64, α = 16, and LoRA dropout of 0.1 for models up to 13B and 0.05 for 33B and 65B models.
## We employ the paged AdamW optimizer with a beta2 value of 0.999, and a learning rate of 2e-4 for models up to 13B and 1e-4 for 33B and 65B models,
## limiting the maximum gradient norm to 0.3 and adopting a constant learning rate strategy. Fine-tuning was executed for 10,000 and 20,000 steps on
## the Alpaca and FLAN v2 datasets, respectively, utilizing a batch size 16.

gpu=""
lora_rank="64"
learning_rate=""
layers=""
extra_name=""
wandb=""
quant_ema_decay=0.99
eval_step_zero=0
hr_lora_r=""
model="huggyllama/llama-7b"
dataset="oasst1"
steps=0
eval_steps=187
save_steps=500
dropout=0.1

qlora="[FILL IN]"

usage() { echo "Usage: $0 -g <gpu_no> -h <hr lora rank csv> -m <model> -d <dataset> -s <steps>[-w <use wanddb> -v <eval_steps> -a <save_steps> -r <dropout> -l <learning rate (og: 0.0002)> -q <quant ema decay> -e <extra name> -t <eval step zero>" 1>&2; exit 1; }

## if w flag is present, set wandb to true
while getopts ":g:h:l:e:a:w:d:t:m:d:q:s:v:a:r" o; do
    case "${o}" in
        g)
            gpu=${OPTARG}
            ;;
        h)
            hr_lora_r=${OPTARG}
            ;;
        l)
            learning_rate=${OPTARG}
            ;;
        e)
            extra_name=${OPTARG}
            ;;
        a)
            save_steps=${OPTARG}
            ;;
        w)
            wandb="--report_to wandb"
            ;;
        q)
            quant_ema_decay=${OPTARG}
            ;;
        t)
            eval_step_zero=1
            ;;
        m)
            model=${OPTARG}
            ;;
        d)
            dataset=${OPTARG}
            ;;
        s)
            steps=${OPTARG}
            ;;
        v)
            eval_steps=${OPTARG}
            ;;
        r)
            dropout=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

echo "lora_rank = ${lora_rank}"
echo "learning_rate = ${learning_rate}"
echo "Recording to wandb? = ${wandb}"
echo "quant_ema_decay = ${quant_ema_decay}"
echo "eval_step_zero = ${eval_step_zero}"
echo "extra_name = ${extra_name}"
echo "hr_lora_r = ${hr_lora_r}"
echo "model = ${model}"
echo "dataset = ${dataset}"
echo "steps = ${steps}"

if [ -z "${hr_lora_r}" ] || [ -z "${gpu}" ] || [ -z "${steps}" ] || [ -z "${model}" ] || [ -z "${dataset}" ]; then
    usage
fi

if [ -z "${extra_name}" ]; then
    name="${model}_${dataset}_l${learning_rate}_${hr_lora_r}"
else
    name="${model}_${dataset}_l${learning_rate}_${hr_lora_r}_${extra_name}"
fi

echo "name = {$name}"
export WANDB_API_KEY="[FILL IN]"

CUDA_VISIBLE_DEVICES=$gpu python $qlora \
    --model_name_or_path $model \
    --output_dir "[FILL IN]"/$name \
    --logging_steps 1 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps $save_steps \
    --save_total_limit 50 \
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
    --dataset $dataset \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps $steps \
    --eval_steps $eval_steps \
    --learning_rate $learning_rate \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout $dropout \
    --weight_decay 0.0 \
    --quant_ema_decay $quant_ema_decay \
    --eval_step_zero $eval_step_zero \
    ${hr_lora_r:+--hr_lora_r $hr_lora_r} \
    --seed 0 \
    $wandb \
    --run_name $name \
    --gpu $gpu \
    --use_auth_token True
