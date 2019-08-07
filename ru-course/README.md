# (Automatic) Speech Recognition 

* LET-REMA-LCEX10


## Keyword spotting project

This project is about keyword spotting using convolutional neural networks with transfer learning (Imagenet). 


### Kaggle

* https://www.kaggle.com/c/tensorflow-speech-recognition-challenge

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
