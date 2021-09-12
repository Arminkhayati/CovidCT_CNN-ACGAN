# CovidCT_CNN-
A Simple code to train a CNN to predict label of Covid and Non-Covid CT scan images and an ACGAN to generate them.<br/>
At first in part 'A' I trained a Resnet50 network that classifies these images with 81% test accuracy.<br/>
Then in part 'B' I trained an ACGAN to generate Covid and Non-Covid CT scan images.<br/>
The classifier from part A, classifies generated images by ACGAN with 79% accuracy.<br/>
Later on I mixed a little bit of generated images with the original train dataset and retrained my network from part A, and got 86% test accuracy on original dataset.<br/>
<br/>
![alt text](https://github.com/Arminkhayati/CovidCT_CNN-/blob/main/ezgif-2-f3146e4642d7.gif?raw=true)


