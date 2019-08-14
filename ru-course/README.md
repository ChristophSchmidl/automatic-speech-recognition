# (Automatic) Speech Recognition 

* LET-REMA-LCEX10


## Keyword spotting project

This project is about keyword spotting using convolutional neural networks with transfer learning (Imagenet). 


### Kaggle

* https://www.kaggle.com/c/tensorflow-speech-recognition-challenge


#### Kaggle's Evaluation

Submissions are evaluated on Multiclass Accuracy, which is simply the average number of observations with the correct label.
Note: There are only 12 possible labels for the Test set: yes, no, up, down, left, right, on, off, stop, go, silence, unknown.
The unknown label should be used for a command that is not one one of the first 10 labels or that is not silence.

**Submission File**

For audio clip in the test set, you must predict the correct label. The submission file should contain a header and have the following format:

```
fname,label
clip_000044442.wav,silence
clip_0000adecb.wav,left
clip_0000d4322.wav,unknown
etc.
```

#### Kernels

- [x] https://www.kaggle.com/davids1992/speech-representation-and-data-exploration
- [x] https://www.kaggle.com/alexozerin/end-to-end-baseline-tf-estimator-lb-0-72
- [x] https://www.kaggle.com/timolee/audio-data-conversion-to-images-eda
- [x] https://www.kaggle.com/alphasis/light-weight-cnn-lb-0-74
- [x] https://www.kaggle.com/ezietsman/simple-keras-model-with-data-generator
	* Replace spectrograms with log spectrograms
	* Why is the generator only using 1415 samples?
	* Change generator so it uses log spectrograms
	* Adjust CNN to MNIST tutorial topology? 
	* Compare the effect of number of epochs on accuracy
- [ ] Look into assignment 5 of ISMI


#### Data distribution

**Original distribution**

Number of classes: 31

| Class              | Count   | Percentage |
|---	             |---	   |---			|
|  stop   		     | 2380    | 0.036770	|
|  yes   		     | 2377    | 0.036723	|
|  seven   		     | 2377    | 0.036723	|
|  zero   		     | 2376    | 0.036708	|
|  up   		     | 2375    | 0.036693	|
|  no   		     | 2375    | 0.036693	|
|  two   		     | 2373    | 0.036662	|
|  four   		     | 2372    | 0.036646	|
|  go   		     | 2372    | 0.036646	|
|  one   		     | 2370    | 0.036615	|
|  six   		     | 2369    | 0.036600	|
|  on   		     | 2367    | 0.036569	|
|  right   		     | 2367    | 0.036569	|
|  nine   		     | 2364    | 0.036523	|
|  down   		     | 2359    | 0.036445	|
|  five   		     | 2357    | 0.036414	|
|  off   		     | 2357    | 0.036414	|
|  three   		     | 2356    | 0.036399	|
|  left   		     | 2353    | 0.036353	|
|  eight   		     | 2352    | 0.036337	|
|  house   			 | 1750    | 0.027037	|
|  dog   			 | 1746    | 0.026975	|
|  marvin   		 | 1746    | 0.026975	|
|  wow   		     | 1745    | 0.026959	|
|  happy   			 | 1742    | 0.026913	|
|  sheila   		 | 1734    | 0.026789	|
|  tree   			 | 1733    | 0.026774	|
|  cat   		     | 1733    | 0.026774	|
|  bird   		     | 1731    | 0.026743	|
|  bed   		     | 1713    | 0.026465	|
| _background_noise_ | 6       | 0.000093	|





**Reduced distribution**

Number of classes: 12

| Class   | Count  | Percentage |
|---	  |---	   |---			|
| unknown |  41039 | 0.634032   |
| stop    |  2380  | 0.036770	|
| yes     |  2377  | 0.036723	|
| up      |  2375  | 0.036693	|
| no      |  2375  | 0.036693	|
| go      |  2372  | 0.036646	|
| right   |  2367  | 0.036569	|
| on      |  2367  | 0.036569	|
| down    |  2359  | 0.036445	|
| off     |  2357  | 0.036414	|
| left    |  2353  | 0.036353	|
| silence |  6     | 0.000093	|


#### Categorical_crossentropy predicition format

```
[[7.7991508e-04 8.3888117e-03 1.6175302e-02 3.7664555e-03 2.5994703e-01
  2.3774782e-03 1.0472519e-03 1.2072919e-04 4.0918142e-02 3.4859773e-02
  6.3102704e-01 5.9204130e-04]]
```

#### Binary_crossentropy predicition format

```
[[9.7345197e-05 5.7157301e-03 8.8743437e-03 1.4735227e-03 1.9038953e-02
  6.0778675e-03 1.9536670e-03 1.2667115e-05 1.7353782e-02 2.8752726e-02
  9.1063923e-01 1.0178681e-05]]
```


#### Transfer learning




- [ ] https://datascience.stackexchange.com/questions/30659/what-are-the-consequences-of-not-freezing-layers-in-transfer-learning
- [ ] https://towardsdatascience.com/transfer-learning-with-convolutional-neural-networks-in-pytorch-dd09190245ce
- [ ] https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a
- [ ] https://github.com/dipanjanS/hands-on-transfer-learning-with-python/tree/master/notebooks#part-ii-essentials-of-transfer-learning
- [ ] https://www.learnopencv.com/keras-tutorial-transfer-learning-using-pre-trained-models/   !!! 
- [ ] https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/  !!!

#### Batch Normlaization

- [ ] https://arxiv.org/abs/1502.03167
- [ ] https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/ (use before or after activation function)
- [ ] https://datascience.stackexchange.com/questions/37889/adding-batch-normalization-layer-to-vgg16-network
- [ ] https://www.learnopencv.com/batch-normalization-in-deep-networks/ !!!

#### Dropout 



#### CNN Architectures

- [ ] https://github.com/jcjohnson/cnn-benchmarks (using ResNet in Keras?)
- [ ] Deep Residual Learning for Image Recognition: https://arxiv.org/abs/1512.03385


#### Model Evaluation

- [ ] https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
- [ ] https://www.kaggle.com/stefanie04736/simple-keras-model-with-k-fold-cross-validation
- [ ] https://gist.github.com/RyanAkilos/3808c17f79e77c4117de35aa68447045




**Model Complexity**

| Model   		| Total params | Trainable params  | Non-trainable params |
|---	  		|---	  	   |---		   		   |---	 	  	  		  |
| Leight CNN	| 723,968 	   | 723,454  	   	   | 514	 	  		  | 
| VGG16  		| 23,676,748   | 23,659,340        | 17,408	 	  		  |
| Inception V3	| 29,185,836   | 29,137,068  	   | 48,768	 	  		  | 
| MNIST			| 55,038,988   | 54,929,868   	   | 109,120	 	  	  | 
| ResNet50  	| 75,182,988   | 75,029,516 	   | 153,472		  	  |	




**Baseline Training**

* Train/Test: 80/20
* Batch size of 32
* Epochs: Epochs it took for the model to converge with patience of 3





| Model   		| Train Accuracy | Train Loss  | Val Accuracy | Val Loss | Training Time (sec) | 
|---	  		|---	  		 |---		   |---	 	  	  |---       |---		 	       |
| MNIST		    | 0.9188 		 | 0.2479      | 0.8816		  | 0.3943	 |	1112.02			   |
| Leight CNN    | 0.9443   	 	 | 0.1711 	   | 0.9400		  |	0.1865	 |  549.69             |






**Training from scratch**

| Model   		| Train Accuracy | Train Loss  | Val Accuracy | Val Loss | Training Time (sec) | 
|---	  		|---	  		 |---		   |---	 	  	  |---       |---		 	       |
| Inception V3	| 0.9338 		 | 0.2238  	   | 0.9427	 	  | 0.1868	 |	2332.23		       |
| VGG16  		| 0.9723   	 	 | 0.0900      | 0.9583	 	  |	0.2011	 |  2530.47            |
| ResNet50  	| 0.9548  	 	 | 0.1421 	   | 0.9445		  |	0.1873	 |  3807.93            |


**Using pre-trained Imagenet models**


| Model   		| Train Accuracy | Train Loss  | Val Accuracy | Val Loss | Training Time (sec) | 
|---	  		|---	  		 |---		   |---	 	  	  |---       |---		 	       |
| Inception V3	| 0.6334  		 | 1.6597  	   | 0.6401	 	  | 1.6433	 |	2184.54			   |
| VGG16  		| 0.9745  	 	 | 0.0912      | 0.9611	  	  |	0.1730 	 |  2541.83            |
| ResNet50  	| 0.9134  	  	 | 0.2991  	   | 0.9252	 	  |	0.3055	 |  3653.67            |


**Training from scratch - First Epoch**

| Model   		| Train Accuracy | Train Loss  | Val Accuracy | Val Loss | 
|---	  		|---	  		 |---		   |---	 	  	  |---       |
| Inception V3	| 0.6539 		 | 1.4176 	   | 0.6778	 	  | 1.1090	 |
| VGG16  		| 0.6519   	 	 | 1.3202      | 0.6906	 	  |	1.3379	 | 
| ResNet50  	| 0.6916  	 	 | 1.2478 	   | 0.6351		  |	5.7467	 | 



**Using pre-trained Imagenet models - First Epoch**


| Model   		| Train Accuracy | Train Loss  | Val Accuracy | Val Loss |
|---	  		|---	  		 |---		   |---	 	  	  |---       |
| Inception V3	| 0.6268  		 | 1.7132  	   | 0.6315	 	  | 2.0153	 |
| VGG16  		| 0.6307  	 	 | 1.4430      | 0.6706	  	  |	1.2011 	 |
| ResNet50  	| 0.6389  	  	 | 1.4214  	   | 0.5931	 	  |	5.1381	 | 






#### Model Training

- [ ] https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/

### Github repositories

* https://github.com/YoavRamon/awesome-kaldi


### Papers

#### Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition

Source: https://arxiv.org/abs/1804.03209


#### Speech Recognition: Keyword Spotting Through Image Recognition (2018)

* Source: https://arxiv.org/abs/1803.03759
* Dataset: Speech Commands Dataset to train and evaluate the model

The authors worked on the task of voice command recognition using the Speech Commands dataset to train and evaluate the model and therefore followed the Tensorflow Speech Recognition Challenge from Kaggle. The authors look into models which are able to run on devices which are limited with regards to memory and computation resources but also unrestricted environments.

"Speech Recognition is the subfield of Natural Language Processing that focuses on understanding spoken natural language. This involves mapping auditory input to some word in a language vocabulary." 

The authors work with a small dataset of 30 words. Their proposed model will learn to identify 10 out of these 30 words and any other words as unknown. Silence is labeled as silence.
 
#### An Experimental Analysis of the Power Consumption of Convolutional Neural Networks for Keyword Spotting
	
Source: https://arxiv.org/pdf/1711.00333.pdf	


#### Convolutional Recurrent Neural Networks for Small-Footprint Keyword Spotting (2017)

Source: https://arxiv.org/abs/1703.05390


#### Honk: A PyTorch Reimplementation of Convolutional Neural Networks for Keyword Spotting

Source: https://arxiv.org/abs/1710.06554	


#### Convolutional neural networks for small-footprint keyword spotting (2015)

Source: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/43969.pdf


#### Small-footprint keyword spotting using deep neural networks (2014)
Source: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/42537.pdf


#### Deep residual learning for small-footprint keyword spotting

Source: https://arxiv.org/pdf/1710.10361.pdf	

#### Transfer learning

* Transfer Learning for Speech Recognition on a Budget
	* https://arxiv.org/pdf/1706.00290.pdf
* Learning and Transferring Mid-Level Image Representations using Convolutional Neural Networks	
	* http://openaccess.thecvf.com/content_cvpr_2014/papers/Oquab_Learning_and_Transferring_2014_CVPR_paper.pdf


### Blogposts

* https://www.tensorflow.org/tutorials/sequences/audio_recognition
* https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5
* https://towardsdatascience.com/how-to-start-with-kaldi-and-speech-recognition-a9b7670ffff6
* http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/

### Papers with Code

* https://paperswithcode.com/task/keyword-spotting
	* https://paperswithcode.com/paper/hello-edge-keyword-spotting-on (20 Nov 2017)
	* https://paperswithcode.com/paper/deep-residual-learning-for-small-footprint (28 Oct 2017) 
	* https://paperswithcode.com/paper/honk-a-pytorch-reimplementation-of (18 Oct 2017)
	* https://paperswithcode.com/paper/efficient-keyword-spotting-using-dilated (19 Nov 2018)
	* https://paperswithcode.com/paper/read-bad-a-new-dataset-and-evaluation-scheme (9 May 2017)
	* https://paperswithcode.com/paper/stochastic-adaptive-neural-architecture (16 Nov 2018)
	* https://paperswithcode.com/paper/speech-commands-a-dataset-for-limited (9 Apr 2018)
	* https://paperswithcode.com/paper/temporal-convolution-for-real-time-keyword (8 Apr 2019)
	* https://paperswithcode.com/paper/whats-cookin-interpreting-cooking-videos (5 Mar 2015)
	* https://paperswithcode.com/paper/javascript-convolutional-neural-networks-for (30 Oct 2018)
	* https://paperswithcode.com/paper/benchmarking-keyword-spotting-efficiency-on (4 Dec 2018)
	* https://paperswithcode.com/paper/an-end-to-end-architecture-for-keyword (28 Nov 2016)

* https://paperswithcode.com/task/small-footprint-keyword-spotting
	* https://paperswithcode.com/paper/deep-residual-learning-for-small-footprint (28 Oct 2017)
	* https://paperswithcode.com/paper/efficient-keyword-spotting-using-dilated (19 Nov 2018)


### Speech corpora

* http://slam.iis.sinica.edu.tw/ldc.htm
* Speech commands dataset: http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz



