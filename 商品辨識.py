# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:23:08 2020

@author: a0988
"""


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout

# Initialising the CNN
model = Sequential()
model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=(256,256,3), 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32,kernel_size=(5,5),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128,kernel_size=(5,5),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(18,activation='sigmoid'))
print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']) 



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,# 隨機錯切換角度
                                   zoom_range = 0.2,# 隨機縮放範圍
                                   horizontal_flip=True, # 一半影象水平翻轉
                                   rotation_range=40, # 角度值，0~180，影象旋轉
                                   width_shift_range=0.2, # 水平平移，相對總寬度的比例
                                   height_shift_range=0.2, # 垂直平移，相對總高度的比例                                  
                                   fill_mode='nearest' # 填充新建立畫素的方法
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training',
                                                 target_size = (256, 256),
                                                 batch_size = 6,                                           
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('test',
                                            target_size = (256, 256),
                                            batch_size = 2,
                                            shuffle=False,
                                            class_mode = 'categorical')
#batch size改變
model.fit_generator(training_set,
                         steps_per_epoch = 30,
                         epochs = 30,
                         validation_data = test_set,
                         validation_steps = 25)





test_set = test_datagen.flow_from_directory('Validation',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            shuffle=False,
                                            class_mode = 'categorical')

result = model.predict(test_set)
test_set.class_indices



#使用resnet進行比較
from tensorflow.keras.applications.resnet import ResNet50
#from tensorflow.keras.applications.resnet import ResNet101
#from tensorflow.keras.applications.inception_v3 import InceptionV3

pre_trained_model = ResNet50(input_shape = (256, 256, 3), # 输入大小
                                include_top = False, # 不要最后的全连接层
                                weights = 'imagenet')
                                #weights = None)
#選擇訓練哪些層
for layer in pre_trained_model.layers:
    layer.trainable = False
    
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import Model
# 为全连接层准备
x = layers.Flatten()(pre_trained_model.output)
# 加入全连接层，这个需要重头训练的
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)   
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)                
# 输出层
x = layers.Dense(18, activation='softmax')(x)           
# 构建模型序列
modell = Model(pre_trained_model.input, x) 

modell.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])    
    
history = modell.fit_generator(
            training_set,           
            steps_per_epoch = 11,
            epochs = 20,
            validation_data = test_set,
            validation_steps = 3,
            #verbose = 2, # callbacks=[callbacks]
            )