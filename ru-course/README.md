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



**Baseline**

* Train/Test: 80/20
* Batch size of 30
* Epochs: Epochs it took for the model to converge with patience of 3

**MNIST model complexity**


Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 95, 157, 32)       832       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 47, 78, 32)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 47, 78, 32)        128       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 45, 76, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 22, 38, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 22, 38, 64)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 53504)             0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 53504)             214016    
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              54789120  
_________________________________________________________________
dropout_2 (Dropout)          (None, 1024)              0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 1024)              4096      
_________________________________________________________________
dense_2 (Dense)              (None, 12)                12300     
=================================================================
Total params: 55,038,988
Trainable params: 54,929,868
Non-trainable params: 109,120


**Leightweight CNN complexity**

Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 99, 161, 1)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 99, 161, 1)        4         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 98, 160, 8)        40        
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 97, 159, 8)        264       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 48, 79, 8)         0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 48, 79, 8)         0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 46, 77, 16)        1168      
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 44, 75, 16)        2320      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 22, 37, 16)        0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 22, 37, 16)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 20, 35, 32)        4640      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 10, 17, 32)        0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 10, 17, 32)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 5440)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               696448    
_________________________________________________________________
batch_normalization_2 (Batch (None, 128)               512       
_________________________________________________________________
dense_2 (Dense)              (None, 128)               16512     
_________________________________________________________________
batch_normalization_3 (Batch (None, 128)               512       
_________________________________________________________________
dense_3 (Dense)              (None, 12)                1548      
=================================================================
Total params: 723,968
Trainable params: 723,454
Non-trainable params: 514


**VGG16 transfer learning complexity**

Layer (type)                 Output Shape              Param #   
=================================================================
vgg16 (Model)                (None, 3, 5, 512)         14714688  
_________________________________________________________________
dropout_1 (Dropout)          (None, 3, 5, 512)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 7680)              0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 7680)              30720     
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              7865344   
_________________________________________________________________
batch_normalization_2 (Batch (None, 1024)              4096      
_________________________________________________________________
dense_2 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
dense_3 (Dense)              (None, 12)                12300     
=================================================================
Total params: 23,676,748
Trainable params: 8,944,652
Non-trainable params: 14,732,096



**VGG16 from scratch complexity**

Layer (type)                 Output Shape              Param #   
=================================================================
vgg16 (Model)                (None, 3, 5, 512)         14714688  
_________________________________________________________________
dropout_1 (Dropout)          (None, 3, 5, 512)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 7680)              0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 7680)              30720     
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              7865344   
_________________________________________________________________
batch_normalization_2 (Batch (None, 1024)              4096      
_________________________________________________________________
dense_2 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
dense_3 (Dense)              (None, 12)                12300     
=================================================================
Total params: 23,676,748
Trainable params: 23,659,340
Non-trainable params: 17,408


| Model   		| Train Accuracy | Train Loss  | Val Accuracy | Val Loss | Training Time (sec) | 
|---	  		|---	  		 |---		   |---	 	  	  |---       |---		 	       |
| MNIST		    | 0.9188 		 | 0.2479      | 0.8816		  | 0.3943	 |	1112.02			   |
| Leight CNN    | 0.9443   	 	 | 0.1711 	   | 0.9400		  |	0.1865	 |  549.69             |




**Training from scratch**

| Model   		| Train Accuracy | Train Loss  | Val Accuracy | Val Loss | Training Time (sec) | 
|---	  		|---	  		 |---		   |---	 	  	  |---       |---		 	       |
| Inception V3	|  		 		 |       	   | 		  	  | 	 	 |				       |
| VGG16  		| 0.9671   	 	 |  0.1082     | 0.9497	 	  |	0.2776	 |  2530.47            |
| ResNet50  	|   	 	 	 |  	   	   | 		  	  |		 	 |                     |





**Using pre-trained Imagenet models**


| Model   		| Train Accuracy | Train Loss  | Val Accuracy | Val Loss | Training Time (sec) | 
|---	  		|---	  		 |---		   |---	 	  	  |---       |---		 	       |
| Inception V3	|  		 		 |       	   | 		  	  | 	 	 |				       |
| VGG16  		| 0.9222  	 	 | 0.2359      | 0.7461	  	  |	1.2772 	 |  1186.45            |
| ResNet50  	|   	 	 	 |  	   	   | 		  	  |		 	 |                     |









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




************ From scratch VGG16 ************


{'val_loss': [1.3379283062007168, 0.4973028878014375, 0.35012419471093037, 0.5347788069741463, 0.2850959170671186, 0.2633146928153047, 0.1942873010043018, 0.37430265727460615, 0.3434643274019073, 0.2011019727191638], 'val_acc': [0.6906456270627063, 0.8699979372937293, 0.909911303630363, 0.9218234323432343, 0.9272896039603961, 0.9219781353135313, 0.9452351485148515, 0.9427083333333334, 0.9431724422442245, 0.958384900990099], 'loss': [1.3202255933318459, 0.5258360299173177, 0.320798707964037, 0.23033373954807912, 0.18443059783902796, 0.3037162405958682, 0.1644459624623373, 0.11864602754588564, 0.10608842736307544, 0.09006048114707055], 'acc': [0.6519213780918728, 0.8252871024734982, 0.8954946996466431, 0.9292181978798587, 0.9437720848056537, 0.9106448763250884, 0.9481007067137809, 0.9643330388692579, 0.968374558303887, 0.9723939929328622]}

Epoch with the highest val acc: 9
Best train_acc: 0.9723939929328622
Best train_loss: 0.09006048114707055
Best val_acc: 0.958384900990099
Best val_loss: 0.2011019727191638
Training time: 2530.5512342453003

************ Pre-trained VGG16 ************

Training time: 2541.8342967033386
Loading best model...
History:
{'val_loss': [1.2011729740958796, 0.7274207058162874, 1.1438769835351716, 0.26832025389343794, 0.1945302869875611, 0.28375428504222755, 0.2636043379298817, 0.28365593774377895, 0.22584547060092283, 0.17303838664930923], 'val_acc': [0.6706373762376238, 0.8460189768976898, 0.7776918316831684, 0.928733498349835, 0.949618399339934, 0.9494121287128713, 0.9498762376237624, 0.9535375412541254, 0.9607570132013201, 0.9611179867986799], 'loss': [1.4430195803145216, 0.578026288052751, 0.2775127211960803, 0.20336113185506616, 0.1609733827389092, 0.13550353136782406, 0.11810706706780137, 0.10774497876812575, 0.09752672969093365, 0.09123584709059061], 'acc': [0.6307641342756184, 0.8073321554770319, 0.9148189045936396, 0.9395318021201413, 0.9525397526501767, 0.9600265017667845, 0.9657464664310954, 0.9699204946996467, 0.9724823321554771, 0.9745362190812721]}
Epoch with the highest val acc: 9
Best train_acc: 0.9745362190812721
Best train_loss: 0.09123584709059061
Best val_acc: 0.9611179867986799
Best val_loss: 0.17303838664930923



************ From scratch ResNet50 ************

Training time: 3807.937078475952
Loading best model...
History:
{'val_loss': [5.746732569370333, 0.9477787557804939, 0.6638708496251122, 0.2533670650401672, 0.2773407854603929, 0.2433992744348144, 0.9533941646078878, 0.23995363217142254, 0.2921666510460019, 0.18738239438952295], 'val_acc': [0.6351072607260726, 0.7451010726072608, 0.7788263201320133, 0.9211530528052805, 0.9144492574257426, 0.9360045379537953, 0.8817037953795379, 0.9317759900990099, 0.9162541254125413, 0.9445132013201321], 'loss': [1.247887090891073, 0.7651383531072535, 0.6836376874533222, 0.32687773995048797, 0.25069151572461895, 0.21477727346425027, 0.44433862687262965, 0.18608289193493632, 0.17942462372445492, 0.14212935330714807], 'acc': [0.6916298586572438, 0.8079063604240283, 0.8357994699646644, 0.8961351590106007, 0.920583038869258, 0.9307420494699646, 0.8691033568904594, 0.9403268551236749, 0.9426236749116608, 0.9548365724381626]}
Epoch with the highest val acc: 9
Best train_acc: 0.9548365724381626
Best train_loss: 0.14212935330714807
Best val_acc: 0.9445132013201321
Best val_loss: 0.18738239438952295


************ Pre-trained ResNet50 ************

Training time: 3653.677728652954
Loading best model...
History:
{'val_loss': [5.138153332294804, 1.3976230613469292, 0.6667361906652797, 1.7655829448412748, 0.6233088257215401, 1.236088703690779, 0.3055559558910702, 0.3009111467509134, 0.5537670461350296, 0.3321740560805296], 'val_acc': [0.5931311881188119, 0.6212355610561056, 0.8325598184818482, 0.7293213696369637, 0.8665944719471947, 0.8344678217821783, 0.9252784653465347, 0.9135726072607261, 0.8738655115511551, 0.9248659240924092], 'loss': [1.4214553795517966, 1.04306427293026, 0.7261600479852185, 0.4516523308025232, 0.3545933911056906, 0.3206416550326052, 0.29915214156725384, 0.20791097713680445, 0.1691111140055154, 0.16992558381016706], 'acc': [0.6389355123674911, 0.712345406360424, 0.8042181978798587, 0.8631183745583039, 0.8941475265017668, 0.9042181978798587, 0.9134275618374559, 0.9359540636042403, 0.9482111307420494, 0.9485203180212014]}
Epoch with the highest val acc: 6
Best train_acc: 0.9134275618374559
Best train_loss: 0.29915214156725384
Best val_acc: 0.9252784653465347
Best val_loss: 0.3055559558910702


************ From scratch Inception V3 ************

Training time: 2332.230175971985
Loading best model...
History:
{'val_loss': [1.1090537390299755, 5.961999742111357, 1.680979469330004, 1.2227466433453482, 1.2598571678002675, 0.5219677761811824, 0.27352184759569736, 0.2570070577385795, 0.28433833547588133, 0.1868882006512211], 'val_acc': [0.6778568481848185, 0.630105198019802, 0.6340243399339934, 0.6721328382838284, 0.6605816831683168, 0.8620565181518152, 0.9181105610561056, 0.9229579207920792, 0.9156353135313532, 0.9427083333333334], 'loss': [1.4176485707608213, 0.8584711846951462, 1.619530827839046, 1.2260530492021002, 0.722917062170514, 0.49077901035652566, 0.33753241518308336, 0.24705905062360392, 0.19980387956746476, 0.22386538439394096], 'acc': [0.6539310954063604, 0.7869037102473498, 0.6318242049469964, 0.6501766784452296, 0.7627429328621909, 0.8460468197879859, 0.8966651943462898, 0.9265680212014135, 0.9400618374558304, 0.9338780918727915]}
Epoch with the highest val acc: 9
Best train_acc: 0.9338780918727915
Best train_loss: 0.22386538439394096
Best val_acc: 0.9427083333333334
Best val_loss: 0.1868882006512211


************ Pre-trained Inception V3 ************

Training time: 2184.5429944992065
Loading best model...
History:
{'val_loss': [2.0153439271174642, 1.7236473415747728, 1.634601202439947, 3.5787416836216113, 1.6655573939332868, 1.6553551375669222, 1.6433027741735917, 14.617252634696834, 1.5690931073903251, 1.6840773519903127], 'val_acc': [0.631549092409241, 0.6373762376237624, 0.6346947194719472, 0.6324257425742574, 0.6380981848184818, 0.6358807755775577, 0.6401608910891089, 0.0875618811881188, 0.6366542904290429, 0.6357776402640264], 'loss': [1.7132237382575395, 1.669777862848747, 1.6518025226812059, 1.6694463715536434, 1.6488565021184645, 1.6525818342454863, 1.6597859418434304, 1.6350550178083008, 1.632096972229624, 1.6127487981698538], 'acc': [0.6268330388692579, 0.6321996466431096, 0.6361086572438163, 0.6321113074204947, 0.6350927561837456, 0.6337455830388693, 0.6334143109540636, 0.6335909893992933, 0.6347835689045936, 0.6363515901060071]}
Epoch with the highest val acc: 6
Best train_acc: 0.6334143109540636
Best train_loss: 1.6597859418434304
Best val_acc: 0.6401608910891089
Best val_loss: 1.6433027741735917
Calculating predictions...