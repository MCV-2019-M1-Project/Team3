#!/bin/bash

for pipe in "dct" "ssim" "color" "lbp"; do
    for root in "data/dataset/week4"; do
        for query in "qsd1_w4"; do
            python main.py --root_folder $root --query $query --pipeline text $pipe --output output_new
        done
    done

    for root in "data/dataset/week3"; do
        for query in "qsd1_w3" "qsd2_w3"; do
            python main.py --root_folder $root --query $query --pipeline text $pipe --output output_new


        done
    done
done
