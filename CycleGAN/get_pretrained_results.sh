#!/bin/sh

FOLDER=$1
MODE_A=$2
MODE_B=$3
DIR=$4

# download given models and dataset
if [ -d ${DIR}${FOLDER}checkpoints/${MODE_A}_pretrained ] 
then
    echo "Model ${MODE_A}_pretrained already downloaded at ${DIR}${FOLDER}checkpoints/${MODE_A}_pretrained."
else
    bash ./scripts/download_cyclegan_model.sh $MODE_A ${DIR}${FOLDER}checkpoints
fi

if [ -d ${DIR}${FOLDER}checkpoints/${MODE_B}_pretrained ] 
then
    echo "Model ${MODE_B}_pretrained already downloaded at ${DIR}${FOLDER}checkpoints/${MODE_B}_pretrained."
else
    bash ./scripts/download_cyclegan_model.sh $MODE_B ${DIR}${FOLDER}checkpoints
fi

if [ -d ${DIR}datasets/${MODE_A}] 
then
    echo "Dataset ${MODE_A} already downloaded at ${DIR}datasets/${MODE_A}."
else
    bash ./datasets/download_cyclegan_dataset.sh $MODE_A ${DIR}datasets
fi

# generate results
python ./test.py --dataroot ${DIR}datasets/$MODE_A/testA --checkpoints_dir ${DIR}${FOLDER}checkpoints --name ${MODE_A}_pretrained --model test --no_dropout --results_dir ${DIR}${FOLDER}results
python ./test.py --dataroot ${DIR}datasets/$MODE_A/testB --checkpoints_dir ${DIR}${FOLDER}checkpoints --name ${MODE_B}_pretrained --model test --no_dropout --results_dir ${DIR}${FOLDER}results