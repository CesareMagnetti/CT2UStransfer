# code to launch ``train.py`` experiments from the cloned CUT repo

DATA_DIR = /vol/biomedic3/hjr119/US_GEN_W_CUT/CUT/datasets/ct2us_hp

# create the directory on bitbucket if it does not exist
mkdir -p /vol/bitbucket/cm1320/CUT/checkpoints/
CHECKPOINT_DIR = /vol/bitbucket/cm1320/CUT/checkpoints/

# launch experiment
python -m visdom.server # for visualisation
python CUT/train.py --dataroot $DATA_DIR --checkpoints_dir $CHECKPOINT_DIR --name experiment1 --model cut --CUT_mode CUT --input_nc 1 --output_nc 1 --display_freq 10