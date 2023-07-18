# This corresponds to the few-shot experiment reported in their paper
# Except that their script doesnt work for some of the data sets
NUM_TRAIN=100
NUM_TEST=1000


for DATA in AG_NEWS DBpedia YahooAnswers 
do
    for SEED in 1 2 3 4 5
    do
        for COMPRESSOR in gzip lz4 hashed_ngram
        do  
            python main_text.py --compressor $COMPRESSOR --seed $SEED --dataset $DATA --num_train $NUM_TRAIN --num_test $NUM_TEST 
        done
    done
done