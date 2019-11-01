#!/bin/bash


for lr in "0.002"; do
    for m in "0.3"; do
        for size in "256"; do
            for val in "val2"; do
                python3 nn.py data/dataset/ --ngpu 2 --batch_size 64 --lr $lr \
                        --epochs 15 --size $size --output output/${lr}_${m}_${size} --margin $m \
                        --test_only --val_set $val
            done
        done
    done
done
