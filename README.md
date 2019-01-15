## Human Activity Recognition using CNN in Keras
This repository contains the code for a small project. The aim of this project is to create a simple Convolutional Neural Network (CNN) based Human Activity Recognition (HAR) system. This system uses the sensor data from a 3D accelerometer for `x`, `y` and `z` axis and recognize the activity of the user e.g. `Walking`, `Jogging`, going `Upstairs` or `Downstairs`, etc.
### Files
The repository contains following files.
* `HAR.py`, Python script file, containing the Keras implementation of the CNN based Human Activity Recognition (HAR) model,
* `actitracker_raw.txt`, Text file containing the dataset used in this experiment,
* `model.h5`, A pretrained model, trained on the training data,
* `evaluate_model.py`, Python script file, containing the evaluation script. This script evaluates the performance of the pretrained netowrk on the provided testData, 
* `testData.npy`, Python data file, containing the testing data used for the evaluation of the available pretrained model,
* `groundTruth.npy`, Python data file, containing the ground truth values for the corresponding outputs for for the test data and
* `README.md`.


### Tools Required

The code in this repository is created using Python 3.6. Furthermore, following libraries are required to run the code provided in this repository:
* `Keras 2.*`
* `Numpy`
* `Matplotlib`
* `Pandas`
* `sklearn`


### Dataset
In these experiments we used the `Actitracker` dataset, released by Wireless Sensor Data Mining (WISDM) lab and can be found at this [[link]](http://www.cis.fordham.edu/wisdm/dataset.php). The data provide in this database is collected from `36` users using a smartphone in there pocket at a sample rate of `20 Samples per second`. The data contains values for acceleration for `x`,`y` and `z` axes, while user performs six different activities in a controlled envoirnment. These activities include 
* `Downstairs`,
* `Jogging`, 
* `Sitting`,
* `Standing`,
* `Upstairs`, and
* `Walking`.

The dataset is not balanced and the distribution of the dataset with respect to the performed activities (class labels) is shown in the figure below.
<p align="center">
<img width="460" height="300" src="https://raw.githubusercontent.com/Shahnawax/HAR-CNN-Keras/master/dataset-distribution.png">
</p>


### Evaluation
A simple CNN based neural network is created using the topology in `HAR.py`. The dataset is splitted into two subgroups, `trainData` and `testData` with the ratio of `80` and `20`% respectively. The training data is further split into training and validation data with the same distribution. The HAR model created in `HAR.py` is then trained on the training data and validated on the validataion data. To evaluate the performance of this network, we write a script "evaluate_model.py". This script uses the `20%` of random samples in the dataset and tests the pretrained CNN model `model.h5`. Furhtermore, this script reports the percentage of the wrong predictions as error and creates a confusion matrix. The results show that the network has an average accuracy of 92.1 %. For further details, the confusion matrix for the HAR on the testData is shown in the figure below:
<p align="center">
<img width="460" height="300" src="https://raw.githubusercontent.com/Shahnawax/HAR-CNN-Keras/master/confusion_matrix.png">
</p>


### Related Problems

The HAR model provided in this work can be further extended to perform the user identification from walking activity. Accelerometer dataset from 22 indivduals can be downloaded from the following [[link]](http://archive.ics.uci.edu/ml/datasets/User+Identification+From+Walking+Activity)
