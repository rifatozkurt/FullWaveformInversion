This code was written by: Divya Shyam Singh (divya.singh@tum.de) and Leon Herrmann (leon.herrmann@tum.de).

Prerequisites:
Please install the following libraries:

PyTorch
pandas
numpy
matplotlib



Directory Contents:
The directory contains Python scripts for the following four methods:

conventionalFWI.py
NNbasedFWI.py
conventionalFWIWithInitialGuess.py
TransferLearningFWI.py



The pretrained model is already included: model_Unet_pretrained_800. (Unet model pretrained on 800 samples for 100 epochs)



Case Studies:
The case studies used in the paper are in data_casestudy.zip. Running the scripts for each method will produce the results shown in the paper.



Data Generation:
It is possible to generate your own training and testing data using GenerateMeasurementDataTrain.py and GenerateMeasurementDataTest.py.



Pretraining:
Pretraining can be performed using Pretraining.py.