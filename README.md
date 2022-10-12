# Deep learning for LUS Assessment

The code in this repository was used in the master's thesis 'Automatic lung aeration 
assessment for pediatric lung ultrasound imaging'. It includes two deep neural networks developed and trained  to 
automatically interpret lung ultrasound (LUS) images. One model is a recurrent convolutional
neural network (RCNN) that classifies LUS videos into LUS scores. The second model is a generative
adversarial network (GAN) that generates segmentation masks containing important clinical features
based on individual LUS frames. 

## Video-based classification
The file LUS_class_training.py is used to load LUS videos and train the model. The data should be 
organized as follows: 

v_(*LUSscore*)_patient

for instance: 

v_score1_s0001.

All data should have a unique name and must be in the same path. Which videos are used for training and testing should be
specified in a trainlist.txt and testlist.txt file. Data can be split with the traintestsplit.py file. If preferred, data can be augmented
with video_augmentation_code.py. 

After training, the performance of the trained network can be tested with LUS_class_inference.py. The file
contains code to generate confusion matrices for all classes and per class. 

## Frame-based segmentation
Using Video2frame.py frames can be extracted from LUS videos. Segmentation masks should be ordered in a 
different folder than the frames and must be in the correct order. This is crucial to create correct input for the model!
Using data_preparation.py frames and segmentation masks are linked together to provide to the model. 
LUS_segmentation_training.py is then used for model training. The performance of the saved model
can then be tested with LUS_segmentation_inference.py. 


