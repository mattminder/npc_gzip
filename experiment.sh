NUM_ROUNDS=100


for DATA in AG_NEWS SogouNews DBpedia YahooAnswers 20News Ohsumed_single R8 R52 kinnews kirnews swahili filipino
do
    for SEED in 1 2 3 4 5
    do
        for COMPRESSOR in gzip lz4
        do  
            python main_text.py --compressor $COMPRESSOR --num_train $NUM_ROUNDS --seed $SEED --dataset $DATA
        done
    done
done