#!/bin/bash
set -e

# main paths
ROOT_PATH=$PWD
SOURCE_PATH=$PWD/src 
DATA_PATH=$PWD/data
OUT_PATH=$PWD/output

# Step 1: Correct training data

# Step 2: Train model
mkdir -p $OUT_PATH/NaiveBayes
mkdir -p $OUT_PATH/MLP
cd $SOURCE_PATH
#python train.py --model naive_bayes --train_data $DATA_PATH/data.json --save_path $OUT_PATH/NaiveBayes/naive_bayes.pkl
#python train.py --model mlp_bow --train_data $DATA_PATH/data.json --save_path $OUT_PATH/MLP/mlp_bow.pkl 
python train.py --model mlp_tfidf --train_data $DATA_PATH/data.json --save_path $OUT_PATH/MLP/mlp_tfidf.pkl

# Step 3: Test model
#cd $SOURCE_PATH
#python predict.py --model $OUT_PATH/NaiveBayes/naive_bayes.pkl
#python predict.py --model $OUT_PATH/MLP/mlp_bow.pkl
python predict.py --model $OUT_PATH/MLP/mlp_tfidf.pkl
