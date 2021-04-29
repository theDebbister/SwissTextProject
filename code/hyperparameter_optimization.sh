#!/bin/sh
# MAINTAINER: Aashish Agarwal

export PYTHONPATH=./DeepSpeech/training

for i in 0.000001 0.00001
   # 0.0001 0.001 0.01
do
    for d in 0.1 0.25
        #0.35
        # 0.5 0.75
    do
        #rm -rf deepspeech_optimizer/summaries
        #rm -rf deepspeech_optimizer/checkpoints
        trainingData = {"Learning rate": $i, "Dropout rate": $d}
        echo ""
        echo "=================================" >> optimization_log.txt
        printf "Learning rate: %s\nDropout: %s\n" "$i" "$d" >> optimization_log.txt
        printf "lr: %s\ndropout: %s" "$i" "$d"
        python DeepSpeech/DeepSpeech.py --train_files "${1}"/train.csv \
        --dev_files "${1}"/dev.csv --test_files "${1}"/test.csv \
        --alphabet_config_path data/alphabet.txt --scorer data/german-text-corpus/kenlm.scorer \
        --test_batch_size 36 --train_batch_size 24 --dev_batch_size 36 --epochs 1 --learning_rate $i --dropout_rate $d --export_dir models \
        --save_checkpoint_dir deepspeech_optimizer/checkpoints \
        --summary_dir deepspeech_optimizer/summaries \
        --early_stop True
        #--metrics_files "test.txt"
    done
done
