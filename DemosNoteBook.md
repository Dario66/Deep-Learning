
 Demo1:   
 mnistDense.ipynb  


**rete iniziale**  
*xin = Input(shape=(784))*  
*res = Dense(10,activation='softmax')(xin)*  

*mynet = Model(inputs=xin,outputs=res)*  

Model: "model"  

 input_1 (InputLayer)        [(None, 784)]      0                                                                    
 dense (Dense)               (None, 10)         7850        
                                                                   
Total params: 7,850  
Trainable params: 7,850  
Non-trainable params: 0  

*mynet.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])*  
*mynet.fit(x_train,y_train_cat, shuffle=True, epochs=10, batch_size=32,validation_data=(x_test,y_test_cat))*  
  
Epoch 10/10  
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2509 - accuracy: 0.9308 - val_loss: 0.2619 - val_accuracy: 0.9279  



**Modifica migliorativa proposta**  
*xin = Input(shape=(784))*  
*x = Dense(128,activation='relu')(xin)*  
*res = Dense(10,activation='softmax')(x)*  
*mynet2 = Model(inputs=xin,outputs=res)*  

**Model: "model_1"**    

 input_2 (InputLayer)        [(None, 784)]             0           
                                                                   
 dense_1 (Dense)             (None, 128)               100480      
                                                                   
 dense_2 (Dense)             (None, 10)                1290        
                                                                   
Total params: 101,770  
Trainable params: 101,770  
Non-trainable params: 0  

*mynet2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])*  
*mynet2.fit(x_train,y_train_cat, shuffle=True, epochs=10, batch_size=32,validation_data=(x_test,y_test_cat))*  


Epoch 10/10  
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0166 - accuracy: 0.9946 - val_loss: 0.0781 - val_accuracy: 0.9795  


**Exercises**  

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

 input_1 (InputLayer)        [(None, 784)]             0           
                                                                   
 dense (Dense)               (None, 128)               100480      
                                                                  
 dense_1 (Dense)             (None, 50)                6450        
                                                                   
 dense_2 (Dense)             (None, 10)                510         
                                                                   
Total params: 107,440  
Trainable params: 107,440  
Non-trainable params: 0  


mynet2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])  
mynet2.fit(x_train,y_train_cat, shuffle=True, epochs=10, batch_size=32,validation_data=(x_test,y_test_cat))  


Epoch 10/10  
1875/1875 [==============================] - 12s 6ms/step - loss: 0.0174 - accuracy: 0.9938 - val_loss: 0.0923 - val_accuracy: 0.9777  

## Considerazioni e analisi risultati  
Agiudicare dai dati i tempi di elaborazione, aumentando di 1 il numero di strati Densi della rete, sono aumentati.  
Paragonando le ultime epoche per le due elaborazioni:  

model_1 1875/1875 [==============================] - 7s 4ms/step - loss: 0.0166 - accuracy: 0.9946 - val_loss: 0.0781 - val_accuracy: 0.9795  

model_2 1875/1875 [==============================] - 12s 6ms/step - loss: 0.0174 - accuracy: 0.9938 - val_loss: 0.0923 - val_accuracy: 0.9777  
  
il valore di perdita, dopo ogni iterazione, risulta minore per il "model_1", sia in fase di training che in fase di validation, dove invece la differenza  risulta più accentuata, mentre l'accuratezza di entrambi i modelli differiscono di poco.  
A giudicare dai risultati eseguiti una sola volta, il modello migliore è il primo (model_1), aggiungendo uno strato Dense ulteriore di 50 neuroni per la parte deeply Connected sembra peggiorare anche se di poco la qualità e le prestazioni del modello.  


