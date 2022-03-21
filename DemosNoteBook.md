
## Demo1:   
## mnistDense.ipynb  


**rete iniziale**  
*xin = Input(shape=(784))*  
*res = Dense(10,activation='softmax')(xin)*  

*mynet = Model(inputs=xin,outputs=res)*  

Model: "model"  
_______________________________________________  
 Layer (type)                Output Shape              Param #     
=================================================================  
 input_1 (InputLayer)        [(None, 784)]             0           
                                                                   
 dense (Dense)               (None, 10)                7850        
                                                                   
=================================================================  
Total params: 7,850  
Trainable params: 7,850  
Non-trainable params: 0  

*mynet.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])*  
*mynet.fit(x_train,y_train_cat, shuffle=True, epochs=10, batch_size=32,validation_data=(x_test,y_test_cat))*  
  
Epoch 1/10  
1875/1875 [==============================] - 9s 3ms/step - loss: 0.4714 - accuracy: 0.8756 - val_loss: 0.3093 - val_accuracy: 0.9148  
Epoch 2/10  
1875/1875 [==============================] - 7s 4ms/step - loss: 0.3039 - accuracy: 0.9152 - val_loss: 0.2808 - val_accuracy: 0.9213  
Epoch 3/10  
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2837 - accuracy: 0.9205 - val_loss: 0.2730 - val_accuracy: 0.9227  
Epoch 4/10  
1875/1875 [==============================] - 7s 4ms/step - loss: 0.2729 - accuracy: 0.9237 - val_loss: 0.2699 - val_accuracy: 0.9254  
Epoch 5/10  
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2669 - accuracy: 0.9257 - val_loss: 0.2681 - val_accuracy: 0.9254  
Epoch 6/10  
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2619 - accuracy: 0.9278 - val_loss: 0.2662 - val_accuracy: 0.9281  
Epoch 7/10  
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2579 - accuracy: 0.9285 - val_loss: 0.2648 - val_accuracy: 0.9284  
Epoch 8/10  
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2555 - accuracy: 0.9297 - val_loss: 0.2631 - val_accuracy: 0.9275  
Epoch 9/10  
1875/1875 [==============================] - 7s 4ms/step - loss: 0.2529 - accuracy: 0.9303 - val_loss: 0.2666 - val_accuracy: 0.9256  
Epoch 10/10  
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2509 - accuracy: 0.9308 - val_loss: 0.2619 - val_accuracy: 0.9279  




**Modifica migliorativa proposta**  
*xin = Input(shape=(784))*  
*x = Dense(128,activation='relu')(xin)*  
*res = Dense(10,activation='softmax')(x)*  
*mynet2 = Model(inputs=xin,outputs=res)*  

**Model: "model_1"**    
_________________________________________________________________  
 Layer (type)                Output Shape              Param #     
=================================================================  
 input_2 (InputLayer)        [(None, 784)]             0           
                                                                   
 dense_1 (Dense)             (None, 128)               100480      
                                                                   
 dense_2 (Dense)             (None, 10)                1290        
                                                                   
=================================================================  
Total params: 101,770  
Trainable params: 101,770  
Non-trainable params: 0  

*mynet2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])*  
*mynet2.fit(x_train,y_train_cat, shuffle=True, epochs=10, batch_size=32,validation_data=(x_test,y_test_cat))*  

Epoch 1/10  
1875/1875 [==============================] - 8s 4ms/step - loss: 0.2603 - accuracy: 0.9267 - val_loss: 0.1453 - val_accuracy: 0.9579  
Epoch 2/10  
1875/1875 [==============================] - 7s 4ms/step - loss: 0.1112 - accuracy: 0.9676 - val_loss: 0.1035 - val_accuracy: 0.9678  
Epoch 3/10  
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0769 - accuracy: 0.9765 - val_loss: 0.0861 - val_accuracy: 0.9733  
Epoch 4/10  
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0577 - accuracy: 0.9823 - val_loss: 0.0830 - val_accuracy: 0.9741  
Epoch 5/10  
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0447 - accuracy: 0.9861 - val_loss: 0.0860 - val_accuracy: 0.9745  
Epoch 6/10  
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0358 - accuracy: 0.9885 - val_loss: 0.0789 - val_accuracy: 0.9759  
Epoch 7/10  
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0287 - accuracy: 0.9908 - val_loss: 0.0777 - val_accuracy: 0.9784  
Epoch 8/10    
1875/1875 [==============================] - 7s 3ms/step - loss: 0.0233 - accuracy: 0.9927 - val_loss: 0.0711 - val_accuracy: 0.9796  
Epoch 9/10  
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0190 - accuracy: 0.9943 - val_loss: 0.0751 - val_accuracy: 0.9783  
Epoch 10/10  
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0166 - accuracy: 0.9946 - val_loss: 0.0781 - val_accuracy: 0.9795  



 
Exercises  

   1- Add additional Dense layers and check the performance of the network  
   2- Replace 'relu' with different activation functions  
   3-Adapt the network to work with the so called sparse_categorical_crossentropy  
   4- the fit function return a history of training, with temporal sequences for all different metrics. Make a plot.  


**1-**   
  
xin = Input(shape=(784))  
x = Dense(128,activation='relu')(xin)  
y = Dense(50,activation='relu')(x)  
res = Dense(10,activation='softmax')(y)  
mynet2 = Model(inputs=xin,outputs=res)  


**Model: "model_2"**  
_________________________________________________________________  
 Layer (type)                Output Shape              Param #     
=================================================================  
 input_1 (InputLayer)        [(None, 784)]             0           
                                                                   
 dense (Dense)               (None, 128)               100480      
                                                                  
 dense_1 (Dense)             (None, 50)                6450        
                                                                   
 dense_2 (Dense)             (None, 10)                510         
                                                                   
=================================================================  
Total params: 107,440  
Trainable params: 107,440  
Non-trainable params: 0  


mynet2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])  
mynet2.fit(x_train,y_train_cat, shuffle=True, epochs=10, batch_size=32,validation_data=(x_test,y_test_cat))  


Epoch 1/10  
1875/1875 [==============================] - 14s 6ms/step - loss: 0.2439 - accuracy: 0.9281 - val_loss: 0.1142 - val_accuracy: 0.9656  
Epoch 2/10  
1875/1875 [==============================] - 11s 6ms/step - loss: 0.1036 - accuracy: 0.9682 - val_loss: 0.0989 - val_accuracy: 0.9692  
Epoch 3/10  
1875/1875 [==============================] - 12s 6ms/step - loss: 0.0721 - accuracy: 0.9779 - val_loss: 0.0921 - val_accuracy: 0.9713  
Epoch 4/10  
1875/1875 [==============================] - 13s 7ms/step - loss: 0.0550 - accuracy: 0.9820 - val_loss: 0.0786 - val_accuracy: 0.9765  
Epoch 5/10  
1875/1875 [==============================] - 11s 6ms/step - loss: 0.0418 - accuracy: 0.9865 - val_loss: 0.0842 - val_accuracy: 0.9757  
Epoch 6/10  
1875/1875 [==============================] - 11s 6ms/step - loss: 0.0332 - accuracy: 0.9893 - val_loss: 0.0984 - val_accuracy: 0.9736  
Epoch 7/10  
1875/1875 [==============================] - 10s 5ms/step - loss: 0.0284 - accuracy: 0.9906 - val_loss: 0.0895 - val_accuracy: 0.9761  
Epoch 8/10  
1875/1875 [==============================] - 11s 6ms/step - loss: 0.0239 - accuracy: 0.9922 - val_loss: 0.0935 - val_accuracy: 0.9751  
Epoch 9/10  
1875/1875 [==============================] - 12s 6ms/step - loss: 0.0216 - accuracy: 0.9930 - val_loss: 0.0876 - val_accuracy: 0.9768  
Epoch 10/10  
1875/1875 [==============================] - 12s 6ms/step - loss: 0.0174 - accuracy: 0.9938 - val_loss: 0.0923 - val_accuracy: 0.9777  

## Considerazioni e analisi risultati  
Agiudicare dai dati i tempi di elaborazione, aumentando di 1 il numero di strati Densi della rete, sono aumentati.  
Paragonando le ultime epoche per le due elaborazioni:  

model_1 1875/1875 [==============================] - 7s 4ms/step - loss: 0.0166 - accuracy: 0.9946 - val_loss: 0.0781 - val_accuracy: 0.9795  

model_2 1875/1875 [==============================] - 12s 6ms/step - loss: 0.0174 - accuracy: 0.9938 - val_loss: 0.0923 - val_accuracy: 0.9777  
  
il valore di perdita, dopo ogni iterazione, risulta minore per il "model_1", sia in fase di training che in fase di validation, dove invece la differenza  risulta più accentuata, mentre l'accuratezza di entrambi i modelli differiscono di poco.  
A giudicare dai risultati eseguiti una sola volta, il modello migliore è il primo (model_1), aggiungendo uno strato Dense ulteriore di 50 neuroni per la parte deeply Connected sembra peggiorare anche se di poco la qualità e le prestazioni del modello.  


