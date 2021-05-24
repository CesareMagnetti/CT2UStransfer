python -m visdom.server # start server for visualisation

# ==== 1st experiment ====

# CycleGAN with standard configuration
python CycleGAN/train.py --dataroot /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp --checkpoints_dir /vol/bitbucket/cm1320/CT2US/checkpoints/ --name CycleGAN_standard --model cycle_gan --input_nc 1 --output_nc 1 --batch_size 2 --cycle_loss L1 --save_epoch_freq 10

# CUT with standard configuration
python CUT/train.py --dataroot /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp --checkpoints_dir /vol/bitbucket/cm1320/CT2US/checkpoints/ --name CUT_standard --model cut --CUT_mode CUT --input_nc 1 --output_nc 1 --batch_size 2 --save_epoch_freq 10

# ==== 2nd experiment ====

# CycleGAN without identity loss
python CycleGAN/train.py --dataroot /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp --checkpoints_dir /vol/bitbucket/cm1320/CT2US/checkpoints/ --name CycleGAN_noIdtLoss --model cycle_gan --input_nc 1 --output_nc 1 --batch_size 2 --cycle_loss L1 --save_epoch_freq 10 --lambda_identity 0

# CUT without identity loss
python CUT/train.py --dataroot /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp --checkpoints_dir /vol/bitbucket/cm1320/CT2US/checkpoints/ --name CUT_noIdtLoss --model cut --CUT_mode FastCUT --input_nc 1 --output_nc 1 --batch_size 2 --save_epoch_freq 10

# ==== 3rd experiment ====

# use LPIPS as a cycle consistency loss both for the model with and without the idt loss
python CycleGAN/train.py --dataroot /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp --checkpoints_dir /vol/bitbucket/cm1320/CT2US/checkpoints/ --name CycleGAN_LPIPS --model cycle_gan --input_nc 1 --output_nc 1 --batch_size 2 --cycle_loss LPIPS --save_epoch_freq 10
python CycleGAN/train.py --dataroot /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp --checkpoints_dir /vol/bitbucket/cm1320/CT2US/checkpoints/ --name CycleGAN_LPIPS_noIdtLoss --model cycle_gan --input_nc 1 --output_nc 1 --batch_size 2 --cycle_loss LPIPS --save_epoch_freq 10 --lambda_identity 0

# ==== 4th experiment ====

# CUT with standard configuration and removed layer 0 from PatchNCE loss
python CUT/train.py --dataroot /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp --checkpoints_dir /vol/bitbucket/cm1320/CT2US/checkpoints/ --name CUT_standard_noLayer0 --model cut --CUT_mode CUT --input_nc 1 --output_nc 1 --batch_size 2 --save_epoch_freq 10 --nce_layers 4,8,12,16
# CUT without identity loss and removed layer 0 from PatchNCE loss
python CUT/train.py --dataroot /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp --checkpoints_dir /vol/bitbucket/cm1320/CT2US/checkpoints/ --name CUT_noIdtLoss_noLayer0 --model cut --CUT_mode FastCUT --input_nc 1 --output_nc 1 --batch_size 2 --save_epoch_freq 10 --nce_layers 4,8,12,16