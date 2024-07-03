#!/bin/bash
path=/thayerfs/home/f004h3t/Workspaces/multi-modal-generative-ai/HR-VQLoRA-Quantized-Hierarchical-Residual-Learning-of-Low-Rank-Adaptors/submodules/qlora-hr-vqlora

# tmux new-session -d -c $path "bash scripts/tiny.sh -g1 -r4 -c16 -l0.001; bash"
# tmux new-session -d -c $path "bash scripts/tiny.sh -g2 -r4 -c64 -l0.001; bash"
# tmux new-session -d -c $path "bash scripts/tiny.sh -g3 -r8 -c64 -l0.001; bash"
tmux new-session -d -c $path "bash; scripts/tiny.sh -g4 -r4 -c16 -l0.001 -e noadd; bash"
tmux new-session -d -c $path "bash; scripts/tiny.sh -g5 -r4 -c64 -l0.001 -e noadd; bash"
