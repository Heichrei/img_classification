# img_classification
Image Classification using CNN

As humans, we are gifted by God with our vision. Our everyday life will require us to see a lot of things. With that vision, we are able to discern between one object to another and in just a very short moment we can classify things subsconciously. But machines are not like us, they are not gifted with natural vision, but they can be ‘taught’ to ‘see’ and classify images. 

In computer science, it is called Computer Vision. There are many approaches in Computer Vision to classify images, but in this use case we will be using a Deep Learning algorithm, known as Convolutional Neural Network (CNN). CNN has been known as the suitable neural network algorithm for imagery, and as the name suggests it works on the convolution principles. These are the tools used in this use case to generate CNN algorithms such as: Google Colab, Python, and TensorFlow Framework. Google Colab is a notebook-type IDE by Google, it is quite convenient as it is free and offers GPU utilization. 


1. Initialization
First, import the required packages: Pandas, Numpy, Matplotlib, and TensorFlow. Pandas package is used for data manipulation, Numpy is for numerical and scientific computing, Matplotlib for data visualization, and TensorFlow to create deep learning framework.



Common machine learning models require data as their input, and through the learning process the machine will be able to identify the logic (weight) behind the input data and target data. The model is trained using Kaggle Image Classification dataset. The dataset consisted of 35,000 images throughout four different categories: architecture; art and culture; foods and drinks; and travel. The dataset consisted of training data, validation data, and testing data. There are 8000 training images for each category.



2. Create Generator
Next we have to create a generator that could feed the training images and validation images into our deep learning model. 




















3. Create Model
The next part is the core of the whole process. We use TensorFlow sequential to create our connected layers. We use three convolutional layers with various sizes of (64x3x3), (128x128x3), and (256x3x3) to get the information of the image. Next we flatten those convoluted image and feeds it into three hidden layers and lastly one output layer consisted of 4 neuron because we have 4 classes of images to classify.




4. Training The Model
Next we train the model into 3 epochs and 176 batch size. We also save the model in case the we face Colab runtime timeout.


After the model has been trained we have to observe the performance of the model using matplotlib to see how good the accuracy and loss throughout each epoch.

























From the observation the model has mediocre performance as the accuracy is not too high and the loss is still high. However if we retrain the model and tune the hyperparameters we could achieve better results.

5. The Prediction

Now we use the model to predict test images. There are ten images to be predicted. First we load each image into the model to be predicted and print the vector value for each image prediction. 

































There are four classes, so there are also four vector representations. [1,0,0,0] means the image belongs to architecture class, [0,1,0,0] means the image belongs to architecture class, [0,0,1,0] means the image belongs to architecture class, [0,0,0,1] means the image belongs to architecture class.

If we observe the image manually, we can consider all of them belong to art and travel class. However the prediction of the model is not that accurate, this is due to the low performance of the model.
