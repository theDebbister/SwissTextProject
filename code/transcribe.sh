#!/bin/bash
for i in /dataMount/School/UZH/MSc/uzhGit/sem04/ml4nlp2/project/localEnv/data/data_swisstext/clips/audio/*.wav ; do
    echo "$(basename $i)"
    deepspeech --model models/output_graph.pb --scorer data/german-text-corpus/kenlm.scorer --audio $i
done
