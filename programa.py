#----------------------solo para uso de google colab---------------------
import numpy as np
from google.colab import drive 
drive.mount('/content/drive')

#----------------------------importar libretias---------------------------
import tensorflow as tf
from tensorflow import keras
#import tensorflow.compat.v1 as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation,Input
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import optimizers

import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd 
import seaborn as sn 
import os

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, classification_report 

#-------------------------------inicio----------------------------------
#numero que identifica a la formula
#para todas lasbases de datos estudiadas 3,4,6,13,14

formulas=[3,4,6,13,14]

#cantidad de neuronas calculadas para cada formula
#BC [8,6,2,360,8160]
#CN [6,5,2,170,2376]
#CT [7,5,2, 47, 667]
#DL [6,5,2,122,1342]
#LC [11,7,2,99,4178]
#LM [8,6,2,106,2376]
#OT [13,7,2,86,5051]
#PT [10,7,2,133,4200]

neurona=[8,6,2,360,8160] 


for x1 in range (len(formulas)):
  for y1 in range (10): #numero de repeticiones para cada formula.
    repeticion=y1
    formula=formulas[x1]
    neuronas=neurona[x1]
    tf.keras.backend.clear_session() 

    #CARGA DE ARCHIVOS
    data = np.genfromtxt('/content/drive/My Drive/Hector/LugMichigan/lung-Michigan_Datos02.txt')
    print(data)
    clases= np.genfromtxt('/content/drive/My Drive/Hector/LugMichigan/lung-Michigan_Clases.txt')
    print(clases)

    #VALIDACION HOLD ON
    x_train, x_test, y_train, y_test = train_test_split(data,clases,test_size=0.3,random_state=42)
    print(x_train.shape,x_test.shape)

    #AUNTOENCODER
    dim_entrada = x_train.shape[1]
    capa_entrada= Input(shape=(dim_entrada,))
    encoder= Dense(neuronas, activation='sigmoid')(capa_entrada)
    decoder= Dense(dim_entrada, activation= 'sigmoid') (encoder)
    autoencoder = Model(inputs=capa_entrada,outputs=decoder)
    encoder1= Model(inputs=capa_entrada, outputs=encoder)
    sgd= SGD(lr=0.01)
    autoencoder.compile(optimizer='sgd',loss= 'mse')
    autoencoder.fit(x_train,x_train, epochs= 500, batch_size=1000, verbose=1)
    x_pred= autoencoder.predict(x_test)
    ecm=np.mean(np.power(x_test-x_pred,2),axis=1)
    print("Error cuadratico medio:")
    print(np.mean(ecm))

    #EXTRACCION DE DATOS DE AUTOENCODER
    x_train_red= encoder1.predict(x_train)
    y_train_red= y_train
    x_test_red= encoder1.predict(x_test)
    y_test_red= y_test

    #RED NEURONAL ARTIFICIAL
    model = Sequential()
    model.add(Dense(13,input_shape=(x_train_red.shape[1],),activation='relu'))
    model.add(Dense(7,activation='relu'))
    model.add(Dense(5,activation='relu'))
    model.add(Dense(2,activation='softmax'))  
    adam=optimizers.Adam(lr=.001)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    snn=model.fit(x_train_red,y_train_red,batch_size=10,nb_epoch=500,shuffle=True)
    evaluation = model.evaluate(x_test_red,y_test_red,batch_size=10,verbose=1)
    snn_pred = model.predict(x_test_red, batch_size=100) 
    snn_predicted = np.argmax(snn_pred, axis=1)
    snn_cm = confusion_matrix(y_test_red, snn_predicted) 
    snn_cmN= np.zeros((len(snn_cm),len(snn_cm)))
    for i in range(len(snn_cm)):
        total=0
        for k in range(len(snn_cm)):
            total=total+snn_cm[i][k]
            total=total.astype(float)
        for j in range(len(snn_cm)):
            snn_cmN[i][j]=(snn_cm[i][j]/total)
    snn_report = classification_report(y_test_red, snn_predicted)
    print(snn_report)
    snn_df_cm = pd.DataFrame(snn_cmN, range(2),range(2)) 
    fig=plt.figure(figsize = (20,8)) 
    sn.set(font_scale=1.4) #for label size 
    sn.heatmap(snn_df_cm, annot=True,annot_kws={"size": 12}) # font size 

    #GUARDAR ARCHIVOS

    info=open('/content/drive/My Drive/Hector/LugMichigan/Reporte'+str(formula)+'_'+str(repeticion)+'.txt','w')
    info.write(snn_report)
    info.close()

    plt.savefig('/content/drive/My Drive/Hector/LugMichigan/MatrizEntrenamiento'+str(formula)+'_'+str(repeticion) , bbox_inches='tight')
