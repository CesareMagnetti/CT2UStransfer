# code to launch ``train.py`` experiments from the cloned CycleGAN repo

# create the directory on bitbucket if it does not exist
mkdir -p /vol/bitbucket/cm1320/CycleGAN/checkpoints/

# launch experiment
#python -m visdom.server # for visualisation
python CycleGAN/train.py --dataroot /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp --checkpoints_dir /vol/bitbucket/cm1320/CycleGAN/checkpoints/ --name LPIPScycle_noIdentity --model cycle_gan --input_nc 1 --output_nc 1 --batch_size 1 --cycle_loss LPIPS --save_epoch_freq 10 --n_epochs 100 --n_epochs_decay 100 --lambda_identity 0 --epoch 190