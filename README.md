# Team3

### Week 1

#### Task 1: Compute image descriptors

We compute the histogram of 256 bins of the images using different color spaces

* RGB
* XYZ
* CIELAB
* CIELUV
* HSV
* HLS
* YCrCb
* Grayscale

#### Task 2: Similarity measures

This are the similarity measures implemented in our code

* Euclidean distance
* L1 distance
* X^2 distance
* Histogram intersection
* Hellinger kernel
* Kullback-Leibler divergence
* Jensen-Shannon divergence


#### Task 3: Compute similarities to museum images for images in QS1

We have tested for all different color spaces and similarity metrics.

#### Task 4: Evaluation QS1

For the query set 1 we have obtained an MAP@10 factor of 0.519 using RGB colorspace and histogram intersection

#### Task 5: Remove back ground for QS2

To remove the background for images in query set 2 we have decided to look at the average value of the pixels at different depths from the boundaries of the global image. Those are the values we have used to segmentate our image. We have achieved a precision of 0.84, recall of 0.87, f1 score of 0.85.

#### Task 6: Evaluation QS2

For the query set 2 we have obtained an MAP@10 factor of 0.08

#### Conclusions week 1
The method to remove the background even tough it seems efficient because the precison, F1 and recall scores are quite high, we have a lot of troubles to recognise the image afterwards. We need to improve the background removal for next week.



### Week 2

#### Task 1: Implement 3D / 2D and block and multiresolution histograms

* 1D histogram
* 2D histogram
* 3D histogram
* Multiresolution histogram
* Pyramid histogram

We have also improved the remove background using morphology.
1. Take the saturation channel on HSV
2. Detect edges using Canny edge detection 
3. Flood fill
4. Morphological opening to remove the unwanted false positives arround the painting

We have achieved a precision of 0.95, recall of 0.97, f1 score of 0.96.

#### Task 2: Test query system using QSD2-W1 (from week1) with Task1 descriptors and evaluate results

From 0.08 MAP@10 obtained last week for QSD 1, this week we have 0.331 using the optimal parameters found last week. But with the new descriptors we have reached up to 0.602 using the pyramid histogram with 3 levels.

#### Task3 3: Detect and remove text from images in QSD1-W2
To detect and remove text we have implemented:
1. Detect number of objects and the position of the second. To do so we have counted the objects in the background mask of the image.
2. Apply a binary mask. We have found that the text box has 128 value for the A and B channels in the cieLAB color space.
3. Morphological opening. It removes the unwanted positives in the mask.
4. Find coordinates using the boundingRect function from openCV.

#### Task 4: Evaluate text detection using bounding box
We have achieved a Mean IoU of 0.877 for QSD 1 and 0.903 for QSD 2

#### Task 5: Test query system using query set QSD1-W2 development
Multyresoluion (5,5) has proven to give the best resutlts achieving 0.916 of MAP@10 and 0.900 of MAP@1

#### Task 6: Detect all the paintings , remove background and text, apply retrieval system
Multyresoluion (5,5) has proven to give the best resutlts achieving 0.633 of MAP@10 and also 0.633 of MAP@1


#### Conclusions week 2
For some images, the method we have used does not remove the shadows from the background.

The remove text method we have implemented wouldn’t work if the text box had some color.

