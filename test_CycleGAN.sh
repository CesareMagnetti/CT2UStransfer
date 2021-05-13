# code to launch ``test.py`` experiments from the cloned CycleGAN repo

# create the directory on bitbucket if it does not exist
mkdir -p /vol/bitbucket/cm1320/CycleGAN/test/

# launch experiment
python CycleGAN/test.py --dataroot /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp --checkpoints_dir /vol/bitbucket/cm1320/CycleGAN/checkpoints/ --results_dir /vol/bitbucket/cm1320/CycleGAN/test/ --name experiment1 --model cycle_gan --input_nc 1 --output_nc 1