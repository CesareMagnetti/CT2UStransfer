# code to launch ``test.py`` experiments from the cloned CycleGAN repo
# create the directory on bitbucket if it does not exist
mkdir -p /vol/bitbucket/cm1320/CUT/test/
# launch experiment
python CUT/test.py --dataroot /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp --checkpoints_dir /vol/bitbucket/cm1320/CUT/checkpoints/ --results_dir /vol/bitbucket/cm1320/CUT/test/ --name experiment1 --model cut --CUT_mode CUT --input_nc 1 --output_nc 1