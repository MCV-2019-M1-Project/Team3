#!/bin/bash


for d in "euclidean" "distance_L" "distance_x2" "intersection" "kl_divergence" "js_divergence" "hellinger"; do
    for c in "HSV" "LAB" "Gray" "XYZ" "HLS" "Luv" "YCbCr"; do
        for b in "100" "200" "400" "800" "1000" "1200" "1300" "1500" "1600" "2000"; do
            for s in "--sqrt" ""; do
                for con in "--concat" ""; do
<<<<<<< HEAD
                    python3 evaluator.py data/dataset --dist $d --color $c $s $con\
=======
                    python3 evaluator.py data/dataset --dist $d --color $c $s $con \
>>>>>>> upstream/master
                        --bins $b --output output/${d}_${c}_${s}_${con}_${b}
                done
            done
        done
    done
done
