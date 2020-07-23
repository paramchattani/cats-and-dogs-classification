import tflearn
import cv2
from tqdm import tqdm
import numpy as np 
from random import shuffle 
import tensorflow as tf
import os 
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt

TRAIN_DIR="C:/Users/param/.spyder-py3/train"

TEST_DIR="C:/Users/param/.spyder-py3/test"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMG_SIZE=20 # we need image size to be 60860 block
LR=1e-3 # we need learning rate to be 1/1000

model_name='dogs_vs_cats-{}'.format('5-conv-layers')


def label_img(img):
    # picture name is dog.93.png
    # if we split ['dog','93','png']
    word_label=img.split('.')[-3]
    if word_label=='cat':
        #[1,0] for cat
        return [1,0]
    else:
        return [0,1]
    # [0,1] for dog 
    
    
def create_train_data():
    train_data=[]
    # tqdm is used for cool loading line  
    
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label=label_img(img)
        #we need full path of the image 
        path=os.path.join(TRAIN_DIR,img)
        img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        # image resized to 50*50 and converted tpo grayscale for easy preprocessing 
        train_data.append([np.array(img),np.array(label)])
        # train data is 50*50 featureset + label of cat and dog
        
    shuffle(train_data)
    #np.save('train_data.py',train_data)
    return train_data 


def create_test_data():
    test_data=[]
    for img in tqdm(os.listdir(TEST_DIR)):
        path=os.path.join(TEST_DIR,img)
        #here format of image is 1.png , 2.png .........
        img_num=img.split('.')[0]
        img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        test_data.append([np.array(img),img_num])
    #np.save('test_data.npy',test_data)
    return test_data 

train_data = create_train_data()
#test_data=create_test_data()

# IMG_SIZE,IMG_SIZE,1 is data input style in matrix form 50*50*1
convnet=input_data(shape=[None,IMG_SIZE,IMG_SIZE,1],name='input')

# 2 layers in convolutional neural network one is convolutional layer and other is maxpool layer 
# that picka maximum number from convolutional window  
convnet=conv_2d(convnet,32,2,activation='relu')
''' 32 is the number of nodes in a convolutional layer
2 is apparantly the window size 

activation = relu means that relu=rectified linear means that whatever thing 
below or equals to 0 is considered 0 and what is a positive integer is considered as it is 

'''
maxpool=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,32,2,activation='relu')
maxpool=max_pool_2d(convnet,2)

# using one full connected layer 
convnet=tflearn.fully_connected(convnet,64,activation='relu')
convnet=dropout(convnet,0.8)

# output layer 

convnet=tflearn.fully_connected(convnet,2,activation='softmax')

convnet=regression(convnet,optimizer='adam',learning_rate=LR,loss='categorical_crossentropy',name='target')

model=tflearn.DNN(convnet)


if os.path.exists('{}.meta'.format(model_name)):
    model.load(model_name)
    print('model loaded')

trains=train_data[:-100]

tests=train_data[-100:]

x=np.array([i[0] for i in trains]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
y=[i[1] for i in trains]

test_x=np.array([i[0] for i in tests]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y=[i[1] for i in tests]

tf.reset_default_graph()
model.fit(x,y,n_epoch=5,validation_set=(test_x,test_y),show_metric=True,snapshot_step=500,run_id=model_name)
    
 
test_data=create_test_data()

fig=plt.figure()

'''enumerate returns list of tuples with indexes attached 
('0','a') ('1','b')
'''

# test data 


for num,data in enumerate(test_data[:12]):
    img_num=data[1]
    img_data=data[0]
    # num 0 then 
    y=fig.add_subplot(3,4,num+1) # photo of 3*4
    orig=img_data
    data=img_data.reshape(IMG_SIZE,IMG_SIZE,1)#IMG_SIZE,IMG_SIZE,1 is just a required form 
    model_out=model.predict([data])[0]
    #model out returns list like [1,0] or [0,1]
    # we only need first element , if first element is 1 it is cat else dog
    if np.argmax(model_out)==1:str_label='dog'
    else:str_label='cat'
    #str_label is keyword
    
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
    
        
    