#!/bin/bash
conda activate multimodal2
conda activate mm_latent
cd /thayerfs/home/f004h3t/Workspaces/multi-modal-generative-ai/storage/Workspaces/multimodal4_codebook_output/qlora-hr-vqlora
scripts/tiny.sh -g0 -r64 -c512 -l0.0002 -w1 -elatent
scripts/tiny.sh -g2 -r64 -c1024 -l0.0002 -w1 -elatent
scripts/tiny.sh -g2 -r64 -c256 -l0.0002 -w1 -elatent
scripts/tiny.sh -g3 -r64 -c512 -l0.0002 -w1 -elatent

scripts/tiny.sh -g3 -r64 -c64 -l0.002 -w1 -elatent_ogloss
scripts/tiny.sh -g4 -r64 -c128 -l0.0002 -w1 -elatent_ogloss
scripts/tiny.sh -g5 -r64 -c256 -l0.0002 -w1 -elatent_ogloss


## Base
conda activate multimodal
scripts/tiny.sh -g1 -r64 -w1 -l0.0002 -eqlora_base


scripts/tiny.sh -g3 -r64 -c1024 -l0.0001 -w1 -elatent_ogloss
scripts/tiny.sh -g3 -r64 -c2048 -l0.0001 -w1 -elatent_ogloss


################## 7/22
#veda
scripts/tiny.sh -g1 -r64 -c4096 -n3 -l0.0002 -w1 -elatent_og_loss # to check a larger codebook at n3
scripts/tiny.sh -g6 -r64 -c2048 -n3 -l0.0002 -w1 -d0.90 -elatent_og_loss # to see if peak bc d90
scripts/tiny.sh -g5 -r64 -c2048 -n4 -l0.0002 -w1 -d0.99 -elatent_og_loss # or peak because n4

#jarvis
scripts/tiny.sh -g5 -r64 -c4096 -n4 -l0.0002 -w1 -elatent_og_loss # to check a larger codebook at n4
scripts/tiny.sh -g6 -r64 -c2048 -n4 -l0.0002 -w1 -d0.90 -elatent_ogloss_delaytrain -a"still fwd passes, target: step 1.3k, start_t_c: 1100" # codebook late train
scripts/tiny.sh -g7 -r64 -c2048 -n3 -l0.0002 -w1 -d0.99 -elatent_ogloss_delaytrain -a"still fwd passes, target: step 2.15k, start_t_c: 1950" # codebook late train

################## 7/23
#changelog
- added codebook start var to linear4bit
- added targeting to linear4bit on start codebook
- The previous 2 enable a passthru so the fwd pass of codebook is not run when codebook is not started
- added a codebook_start step parameter in script
- added a eval_step_zero parameter in script

#veda
scripts/tiny.sh -g1 -r64 -c4096 -n3 -l0.0002 -w1 -t1 -elatent_og_loss # to check a larger codebook at n3 # rurun of line 26
scripts/tiny.sh -g2 -r64 -c4096 -n4 -l0.0002 -w1 -d0.90 -elatent_ogloss_delaytrain -a"still fwd passes, target: step 1.3k, start_t_c: 1100" # to confirm codebook train, weird

#jarvis
scripts/tiny.sh -g1 -r64 -c2048 -n3 -l0.0002 -w1 -d0.99 -elatent_ogloss_delaytrain -s1 -t1950 -a"still fwd passes, target: step 2.15k, start_t_c: 1950" # codebook late train

################## 7/25
#changelog
- Fix codebook start step detection

#veda
scripts/tiny.sh -g0 -r64 -c2048 -n3 -l0.0002 -w1 -s950 -elatent_og_loss_delaytrain -a"target: step 1.1k, preempt: 150 steps"
scripts/tiny.sh -g3 -r64 -c2048 -n3 -l0.0002 -w1 -s850 -elatent_og_loss_delaytrain -a"target: step 1.1k, preempt: 250 steps"

#jarvis
scripts/tiny.sh -g0 -r64 -c4096 -n4 -l0.0002 -w1 -s850 -elatent_og_loss_delaytrain -a"target: step 1.1k, preempt: 250 steps"
scripts/tiny.sh -g1 -r64 -c4096 -n4 -l0.0002 -w1 -s950 -elatent_og_loss_delaytrain -a"target: step 1.1k, preempt: 150 steps"

################## 7/26
- Fix codebook again, epoch steps fall off the earth before

#jarvis
scripts/tiny.sh -g0 -r64 -c4096 -n4 -l0.0002 -w1 -s850 -elatent_og_loss_delaytrain -a"target: step 1.1k, preempt: 250 steps"
scripts/tiny.sh -g1 -r64 -c4096 -n4 -l0.0002 -w1 -s950 -elatent_og_loss_delaytrain -a"target: step 1.1k, preempt: 150 steps"
scripts/tiny.sh -g2 -r64 -c2048 -n3 -l0.0002 -w1 -s950 -elatent_og_loss_delaytrain -a"target: step 1.1k, preempt: 150 steps"
scripts/tiny.sh -g3 -r64 -c2048 -n3 -l0.0002 -w1 -s850 -elatent_og_loss_delaytrain -a"target: step 1.1k, preempt: 250 steps"
