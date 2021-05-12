# code to launch ``train.py`` experiments from the cloned CUT repo

# create the directory on bitbucket if it does not exist
mkdir -p /vol/bitbucket/cm1320/CycleGAN/checkpoints/

# launch experiment
python -m visdom.server # for visualisation
python CUT/train.py --dataroot /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp --checkpoints_dir /vol/bitbucket/cm1320/CUT/checkpoints/ --name experiment1 --model cycle_gan --input_nc 1 --output_nc 1 --batch_size 2