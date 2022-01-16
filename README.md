# eyeAware
We devised a web-app that continously track student attention and enables real time monitoring and warning of non-attentive students

## Step 1: Create a model to detect facial pointers
The very first step is to create a model to detect facial pointers such as nose, eyes, and mouth. To do this I just used [FaceScrub](http://vintage.winklerbros.net/facescrub.html) from the Vision & Interaction Group NUS Singapore and [dlib face landmarks](https://github.com/davisking/dlib-models) as a basis to detect automatic facial behavior and analysis. Openface is used because of its extensive training data and real-time performance capability and the application does not need any specialist hardware.

## Step 2: Cleaning the data
Not all camera output qualities are built equally, if the program has lower than a 20% confidence on the detection of facial features it will create a static image of the zoom feed and use [waifu2x](https://github.com/nagadomi/waifu2x) to enable image upscaling. This will be done using a dedicated cloud instance that is able to support CUDA-based SRCNN operations

## Step 3: Preprocessing and Augmentation
Before we can feed our data to train our neural net we first need to do some data normalization and some data augmentation. It turns out that we don't have an equal number of attentive and un-attentive student images which is a problem when training a classifier. To fix this problem we can do some data augment by sampling images from each of the class and applying a random rotation and blur to the image to get more data.

This method can be used to greatly increase the amount of data we have since neural nets need a “large” amount of data to get good results:
```
def rotateImage(img, angle):
    (rows, cols, ch) = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    return cv2.warpAffine(img, M, (cols,rows))
    
    
def loadBlurImg(path, imgSize):
    img = cv2.imread(path)
    angle = np.random.randint(0, 360)
    img = rotateImage(img, angle)
    img = cv2.blur(img,(5,5))
    img = cv2.resize(img, imgSize)
    return img

def loadImgClass(classPath, classLable, classSize, imgSize):
    x = []
    y = []
    
    for path in classPath:
        img = loadBlurImg(path, imgSize)        
        x.append(img)
        y.append(classLable)
        
    while len(x) < classSize:
        randIdx = np.random.randint(0, len(classPath))
        img = loadBlurImg(classPath[randIdx], imgSize)
        x.append(img)
        y.append(classLable)
        
    return x, y

def loadData(img_size, classSize):
    attentive = glob.glob('./attentive/**/*.jpg', recursive=True)
    unattentive = glob.glob('./unattentive/**/*.jpg', recursive=True)
    
    
    imgSize = (img_size, img_size)
    xattentive, yattentive = loadImgClass(attentive, 0, classSize, imgSize)
    xunattentive, yunattentive = loadImgClass(notHotdogs, 1, classSize, imgSize)
    print("There are", len(xattentive), "attentive students")
    print("There are", len(xunattentive), "non-attentive students")
    
    X = np.array(xattentive + yunattentive)
    y = np.array(yattentive + yunattentive)
    
    return X, y
 ```   
To normalize our images we convert them to gray scale and then preform [histogram equalization](https://en.wikipedia.org/wiki/Histogram_equalization)

```
def toGray(images):
    images = 0.2989*images[:,:,:,0] + 0.5870*images[:,:,:,1] + 0.1140*images[:,:,:,2]
    return images

def normalizeImages(images):
    images = (images / 255.).astype(np.float32)
    
    for i in range(images.shape[0]):
        images[i] = exposure.equalize_hist(images[i])
    
    images = images.reshape(images.shape + (1,)) 
    return images

def preprocessData(images):
    grayImages = toGray(images)
    return normalizeImages(grayImages)
```

## Step 4: Building The Neural Net
The model is a convolutional neural networks based on the steering angle model for building self-driving cars built by [comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py)


The model includes ELU layers and dropout to introduce nonlinearity:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 128x128x1 Gray scale image  					| 
| Convolution 8x8     	| 4x4 subsampling 								|
| ELU			      	| 							 					|
| 						|												|
| Convolution 5x5	    | 2x2 subsampling								|
| ELU					|												|
| 						|												|
| Convolution 5x5	    | 2x2 subsampling								|
| Flatten 				| 												|
| Dropout				| .2 dropout probability						|
| ELU					|												|
|						|												|
| Fully connected		| output 512   									|
| Dropout				| .5 dropout probability						|
| ELU					|												|
|						|												|
| Fully connected		| output 2   									|
| Softmax               | output 2                                      |

To actually code this up we will use Keras which is built on top of TensorFlow

```
def kerasModel(inputShape):
    model = Sequential()
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4),border_mode='valid', input_shape=inputShape))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model
```

## Step 5: Training The Neural Net
To train the network we split our data into a tranining set and a test set
```
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
```
Then is as simple to train
```
inputShape = (128, 128, 1)
model = kerasModel(inputShape)
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_train, y_train, nb_epoch=10, validation_split=0.1)
```

## Step 6: The Results
To test the model on the test set we just do
```
metrics = model.evaluate(X_test, y_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))
```

The result is around 70% accuracy in detecting student attentiveness. Given enough time and a bigger dataset we expect that this model will become more accurate
