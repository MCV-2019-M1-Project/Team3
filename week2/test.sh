#!/bin/bash
for h in "1D" "2D" "3D" "multiresolution" "pyramid"; do
  for d in "intersection"  "euclidean"; do
      for c in "HSV" "LAB" "XYZ" "HLS" "Luv" "YCbCr"; do
          for b in "64" "128" "256"; do
              for con in "--concat"; do
                      echo ${h}_${d}_${c}_${con}_${b}
                      python3 evaluator.py --histogram $h   --dist $d --color $c $con \
                          --bins $b --output output/${h}_${d}_${c}_${con}_${b} --save
              done
          done
      done
  done
done
