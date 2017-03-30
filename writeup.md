**Traffic Sign Recognition** 

Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./traffic_images/30.jpg "Traffic Sign 30Km/h"
[image5]: ./traffic_images/children_crossing.jpg "Childrens crossing"
[image6]: ./traffic_images/no_entry.jpg "No Entry"
[image7]: ./traffic_images/no_truck_passing.jpg "No truck passing"
[image8]: ./traffic_images/right_turn.jpg "Right turn"
[image9]: ./examples/traffic_signs_examples.png "Example Data"
[image10]: ./code_images/1.png
[image11]: ./code_images/2.png
[image12]: ./code_images/original.png "Original"
[image13]: ./code_images/5degrees.png "Rotation 5 degrees"
[image14]: ./code_images/minus5degrees.png "Rotation -5 degrees"
[image15]: ./code_images/lenet.png "LeNet"
[image16]: ./code_images/3.png


## Rubric Points


<b>Files Submitted:</b>

| Criteria            |   Meets Specifications                |
|---------------------|---------------------------------------|
| Submission Files    | Traffic_Sign_Classifier_juanlvo.ipynb |



<b>Dataset Exploration:</b>

| Criteria            |   Meets Specifications                |
|---------------------|---------------------------------------|
| Dataset Summary     | the dataset chose for this project is: [German Dataset](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip) this dataset is recomend by Udacity |
|Exploratory Visualization| ![alt text][image9] |


<b>Design and Test a Model Architecture:</b>

<table>
    <tr>
        <th>Criteria</th>
        <th>Meets Specifications</th>
    <tr>
    <tr>
        <td>Preprocessing</td>
        <td>For the preprocessing was used LeNet technique</td>
    </tr>
    <tr>
        <td>Model Architecture</td>
        <td>The model arquitecture is base in 5 Layers: <br/> 
            1. Layer 1: Convolutional network. Input = 32x32x1. Output = 28x28x6. <br/> 
            2. Layer 2: Convolutional network. Output = 10x10x16. <br/> 
            3. Layer 3: Fully Connected. Input = 400. Output = 120. <br/> 
            4. Layer 4: Fully Connected. Input = 120. Output = 84. <br/> 
            5. Layer 5: Fully Connected. Input = 84. Output = 43.<br/> </td>
    </tr>
    <tr>
        <td>Model Training</td>
        <td>The model was trained for 30 Epochs, as well for improve the accuracy of the training there is a dropout in the 3rd Layer, because was overtrain LeNet, the batch size chose was 128 and the optimizer used is Adam, the sigma value for LeNet was modified to 0.05 because help to improve the results of the training.</td>
    </tr>
    <tr>
        <td>Solution Approach</td>
        <td>Using LeNet without anymodification in the dataset was reached 85% of accuracy of the test, but adding to the training dataset the same iamges rotated 5 degrees and -5 degrees help to reach 93% of accuracy which was the minimun need it for meet specifications</td>
    </tr>
</table>



<b>Test a Model on New Images:</b>

<table>
    <tr>
        <th>Criteria</th>
        <th>Meets Specifications</th>
    </tr>
    <tr>
        <td>Acquiring New Images</td>
        <td>The new images (german traffic signs) was found it on google images with the specifications to be 32x32 pixels</td>
    </tr>
    <tr>
        <td>Performance on New Images</td>
        <td>The predictions normally reach 100% of accuracy, in some cases 75% of accuracy</td>
    </tr>
</table>

<b>Suggestions for improvments</b>

The accuracy of this project could be improve with a bigger training dataset, adding more images rotates, modifying the ligth over every image, but with the incresing of the training data set is going to be incresing the time for training, this point is important because if we are using a service like Amazon Web Service this implies an increase of the bill at the end of the month.



---
<b>Writeup / README </b>


[Project code](https://github.com/juanlvo/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_juanlvo.ipynb)

Data Set Summary & Exploration

1. Basic Summary:

* The size of training set is ? 104397
* The size of test set is ? 12630
* The shape of a traffic sign image is ? (32, 32, 3)
* The number of unique classes/labels in the data set is ? 43

![alt text][image10]

2. Exploratory visualization of the dataset: 

Here in the next image you can see where is the data exploration

![alt text][image11]

3. Design and Test a Model Architecture

For preprocessing the image was necessary to incresse the number of images in the training set, the technique selected was rotate images 5 degrees and -5 degrees, here some examples

Original
![alt text][image12]

Rotation 5 degrees
![alt text][image13]

Rotation -5 degrees
![alt text][image14]

The arquitecture use was LeNet, here an image of the explanation of the arquitecture:

![alt text][image15]


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  Output 14x14x6 				|
| Convolution 3x3	    | 2x2 stride,  Output 5x5x16					|
| Fully connected		| Input = 400. Output = 120.				|
| Dropout       		| Dropout of 0.99				|
| Matmul        		| Output = 84.				|
| Matmul        		| Output = 43.				|
 

4. Training and optimizer

For the training was using the optimizer Adam with rate = 0.001

Here the code where can found it all the call for the training

![alt text][image16]

5. Approach

My final model results were:
 * Test Accuracy = 0.937
 * My Data Set Accuracy = 0.800
 * Validation Accuracy = 0.945


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 