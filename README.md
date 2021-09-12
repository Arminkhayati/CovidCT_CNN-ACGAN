# CovidCT_CNN-
A Simple code to train a CNN to predict label of Covid and Non-Covid CT scan images and an ACGAN to generate them.
At first in part 'A' I trained a Resnet50 network that classifies these images with 81% test accuracy.
Then in part 'B' I trained an ACGAN to generate Covid and Non-Covid CT scan images.
The classifier from part A, classifies generated images by ACGAN with 79% accuracy.
Later on I mixed a little bit of generated images with the original train dataset and retrained my network from part A, and got 86% test accuracy on original dataset.



