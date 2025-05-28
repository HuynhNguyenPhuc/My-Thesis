#!/bin/bash

# python test.py --data_dir ./datasets/House3K --save_dir ./test/diffusion --ss_model ./experiments/Diffusion/ss/run_1_diffusion_full --slat_model ./experiments/Diffusion/slat/run_1_diffusion_full

# python evaluate.py --data_dir ./datasets/House3K --test_dir ./test/diffusion --kfid --clip


python test.py --data_dir ./datasets/House3K --save_dir ./test/distillation_time_4_4_768 --ss_model ./experiments/Distillation/ss/run_1_4_4_768 --slat_model ./experiments/Distillation/slat/run_1_4_4_768


# python evaluate.py --data_dir ./datasets/House3K --test_dir ./test/distillation_4_4_768 --kfid --clip



# python test.py --data_dir ./datasets/House3K --save_dir ./test/lora_pt_10 --ss_model ./experiments/LoRA/ss/run_0_patience_10 --slat_model ./experiments/LoRA/slat/run_0_patience_10 --lora True
# python evaluate.py --data_dir ./datasets/House3K --test_dir ./test/lora_pt_10 --kfid --clip


# python demo.py \
# --prompt "A modern minimalist two-story house with a flat roof and large floor-to-ceiling glass facade, featuring clean white walls and subtle natural wood accents. The interior is visible through the glass, showcasing an open-plan layout with sleek wooden flooring, neutral-toned furniture, minimal decor, and built-in shelving. Isolated on a neutral grey background, high-detail architecture, centered perspective, no exterior environment or vegetation." \
# --save_dir ./temp \
# --ss_model ./experiments/LoRA/ss/run_4_new_lr_1e-5_pt_10 \
# --slat_model ./experiments/LoRA/slat/run_4_new_lr_1e-5_pt_10 \
# --lora True