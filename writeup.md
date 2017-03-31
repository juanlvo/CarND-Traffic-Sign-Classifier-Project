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
[image4]: ./traffic_images/9.jpg "Traffic Sign 30Km/h"
[image5]: ./traffic_images/children_crossing.jpg "Childrens crossing"
[image6]: ./traffic_images/no_entry.jpg "No Entry"
[image7]: ./traffic_images/no_truck_passing.jpg "No truck passing"
[image8]: ./traffic_images/right_turn.jpg "Right turn"
[image9]: ./examples/traffic_signs_examples.png "Example Data"
[image10]: ./code_images/1.png
[image11]: ./code_images/2.png
[image12]: ./code_images/6.png "Original"
[image13]: ./code_images/7.png "Rotation 10 degrees"
[image14]: ./code_images/8.png "Rotation -10 degrees"
[image15]: ./code_images/lenet.png "LeNet"
[image16]: ./code_images/3.png
[image17]: ./code_images/14.png
[image18]: ./code_images/15.png



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
        <td>Using LeNet without any modification in the dataset was reached 85% of accuracy of the test, but adding to the training dataset the same iamges rotated 10 degrees and -10 degrees help to reach 93% of accuracy which was the minimun need it for meet specifications</td>
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
        <td>The new images (german traffic signs) was found it on google images with the specifications to be 32x32 pixels but the images were rotated 20 degrees to have a characteristic that make it difficult for the classifier</td>
    </tr>
    <tr>
        <td>Performance on New Images</td>
        <td>The accuracy is not good enough because in the dataset there is no images rotated at 20 degrees, the minimun accuracy was 0% and the maximun was 60%</td>
    </tr>
</table>

<b>Suggestions for improvments</b>

The accuracy of this project could be improve with a bigger training dataset, adding more images rotates with more degrees, modifying the ligth over every image, but with the incresing of the training data set is going to be incresing the time for training, this point is important because if we are using a service like Amazon Web Service this implies an increase of the bill at the end of the month.



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

For preprocessing the image was necessary to incresse the number of images in the training set, the technique selected was rotate images 10 degrees and -10 degrees, here some examples

Original:
![alt text][image12]

Rotation 10 degrees:
![alt text][image13]

Rotation -10 degrees:
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
 * Test Accuracy = 0.945
 * My Data Set Accuracy = 0.600
 * Validation Accuracy = 0.955


6. Test a Model on New Images

For this project was tested the model on 5 new images of German traffic signs found it on internet

Here are five German traffic signs that I found on the web:

![alt text][image4] 

Using softmax the prediction was accurate enough, here we can see the pice of code where is executing the predictions

![alt text][image17]

Here are the results of the prediction:

![alt text][image18]

The model was able to correctly guess 3 of the 5 traffic signs, in this case problem with the guessing of every image is related with the fact of the rotation of 20 degrees, in the training dataset there is no images rotated at 20 degree, so this is making difficult for the algoritm to recognize it.

Analisys of every case:

1. 30 Km/h: As you realize in the 5 posibilities of softmax, the algoritm was not able to detect correcly which image was the correct.

2. Crossing childrens: we got the same situaion as the case before the algoritm was not able to recognize the right sign, but we can realized in the first guess the form of the sign was a triangle, but not the right one.

3. No Entry: in this case softmax was able to recognize the right image.

4. No passing for vehicles over 3.5 metric tons: in this case softmax was able to recognize the right image.

5. Turn right ahead: in this case softmax was able to recognize the right image.

Here we can see the accuracy of the prediction for every image:

<table>
    <tr>
        <th>Image number</th>
        <th>Accuracy</th>
    </tr>
    <tr>
        <td>1</td>
        <td>0%</td>
    </tr>
    <tr>
        <td>2</td>
        <td>0%</td>
    </tr>    
    <tr>
        <td>2</td>
        <td>33,33%</td>
    </tr>   
    <tr>
        <td>3</td>
        <td>50%</td>
    </tr>    
    <tr>
        <td>4</td>
        <td>60%</td>
    </tr>      
</table>

The accuracy of the training (94,5%) vs the accuracy of the new images (between 0% and 60%) show it there is a problem with the dataset of training because the gap of accuracy is too big and at the end the most important is to test the recognition of traffic sign in the real world images for any reason can be rotated, images not was put it in the right sense, images with low ilumination, images hit it by something, images with graffitis or stickers (quite commun nowadays).

<b>Conclusion:</b>

This solution is good enough for complete the objectives of this project but with this kind of results is clear the algorithm need to be train with more images for fit in a real world where is need a higher accuracy.