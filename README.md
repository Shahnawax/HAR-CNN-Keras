## Human Activity Recognition using CNN in Keras
This project is to create a small and simple Convolutional Neural Network for Human Activity Recognition using accelerometer sensor data for x, y and z axis.
### Files
The repository contains
* `HAR.py`, containing the Keras network implementation of the CNN for Human Activity Recognition,
* `actitracker_raw.txt`, the dataset used for this experiment,
* `model.h5`, a pretrained model on the training data,
* `evaluate_model.py`, containing the evaluation script. This script evaluates the performance of the pretrained netowrk, 
* `testData.npy`, containing the testing data used for the evaluation of the model,
* `groundTruth.npy`, containing the ground truth values for the testData outputs and
* `README.md`.
The classification accuracy of the network is approximately 92%.


### Tools Required

Python 3.6 is used during development and following libraries are required to run the code provided in the notebook:
* Keras 2.*
* Numpy
* Matplotlib
* Pandas
* sklearn


### Dataset
We will use Actitracker data set released by Wireless Sensor Data Mining (WISDM) lab. 
The dataset we used for this project was released by Wireless Sensor Data Mining `(WISDM)` Lab and can be found on this [[link]](http://www.cis.fordham.edu/wisdm/dataset.php).
The database has data for  axis `x`,`y` and `z` for user to perform six different activities in controlled envoirnments. The activities include 
* Downstairs,
* Jogging, 
* Sitting,
* Standing,
* Upstairs and
* Walking.

The data is collected from 36 users using a smartphone in their pocket with the 20Hz sampling rate (20 samples per second). The dataset distribution with respect to activities (class labels) is shown in the figure below.
<p align="center">
<img width="460" height="300" src="https://raw.githubusercontent.com/Shahnawax/HAR-CNN-Keras/master/dataset-distribution.png">
</p>


### Evaluation
A network is created using the topology in `HAR.py` and trained on 80% of the data. To evaluate the performance of this network, we write a script "evaluate_model.py". This script uses the 20% of random samples in the dataset and tests the pretrained cnn model `model.h5`. This reports the percentage of the wrong predictions as error and creates a confusion matrix. The results shows that the network has an accuracy of 92.1 %. The confusion matrix is shown below
<p align="center">
<img width="460" height="300" src="https://raw.githubusercontent.com/Shahnawax/HAR-CNN-Keras/master/confusion_matrix.png">
</p>


### Related Problems

User identification from walking activity. Accelerometer dataset from 22 indivduals can be downloaded from the following [[link]](http://archive.ics.uci.edu/ml/datasets/User+Identification+From+Walking+Activity)

