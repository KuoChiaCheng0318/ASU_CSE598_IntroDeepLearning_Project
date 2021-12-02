# ASU_CSE598_IntroDeepLearning_Project

This repository: https://github.com/KuoChiaCheng0318/ASU_CSE598_IntroDeepLearning_Project  contains my code for ASU_CSE598_IntroDeepLearning_Project
<br /><br />
dog-breed-identification_BreedHistorgram.py: Plots the histogram of the frequency of every dog breeds.<br /><br />
dog-breed-identification_confusionMatrix.py: Plots the Confusion_matrix of the testing result of the main success program.<br /><br />
dog-breed-identification_CudaOutofMemory.py: The program trying to use HW4a model which can input 224X224 image, but cannot use GPU due to Cuda out of memory. I then tried to use cpu instead.<br /><br />
dog-breed-identification_DummyClassifier.py: The program which do the baseline of DummyClassifier.<br /><br />
**dog-breed-identification_MainSuccess.py: The main program referenced from https://techvidvan.com/tutorials/dog-breed-classification/ website, which uses ResNet50V2, and works well on training and testing.<br /><br />**
dog-breed-identification_Origin.py: The original code from https://techvidvan.com/tutorials/dog-breed-classification/ website.<br /><br />
dog-breed-identification_ShowErrorImage.py: The program which uses the MainSuccess program, but can also output the error predicted images.<br /><br />
dog-breed-identification_ShowInputImage.py: The program which uses matplotlib to show the input images.<br /><br />
dog-breed-identification_SimpleCNNoverfit.py: The program which uses a simple CNN model referenced from https://www.kaggle.com/androbomb/using-cnn-to-classify-images-w-pytorch website. The testing result was not good, but have good performance on training (Overfit).<br /><br />
dog-breed-identification_useHW4b.py: The program which uses HW4b model to train and test. Because it only input 28X28 grayscale images, the result is not good.<br /><br />
LogDifferentEpochs.txt: The Log file of the CNN model which overfits on testing. I tried different epochs, to see when should I stop training.<br /><br />
