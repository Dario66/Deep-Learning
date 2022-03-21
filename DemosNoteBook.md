
 Demo1:   
 mnistDense.ipynb  


**rete iniziale**  
*xin = Input(shape=(784))*  
*res = Dense(10,activation='softmax')(xin)*  

*mynet = Model(inputs=xin,outputs=res)*  
     
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


**Esercizio 1**    
  
xin = Input(shape=(784))  
x = Dense(128,activation='relu')(xin)  
y = Dense(50,activation='relu')(x)  
res = Dense(10,activation='softmax')(y)  
mynet2 = Model(inputs=xin,outputs=res)  


**Model: "model_2"**  
                                                                   
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

Modello migliore Model_1:
model_1 1875/1875 [==============================] - 7s 4ms/step - loss: 0.0166 - accuracy: 0.9946 - val_loss: 0.0781 - val_accuracy: 0.9795  

**Esercizio 2**    

Specificando una funzione di attivazione diversa per il primo strato, per esempio "tanh":  

xin = Input(shape=(784))  
x = Dense(128,activation='tanh')(xin)  
res = Dense(10,activation='softmax')(x)  
  
Epoch 10/10  
1875/1875 [==============================] - 9s 5ms/step - loss: 0.0147 - accuracy: 0.9964 - val_loss: 0.0830 - val_accuracy: 0.9772  

si puo notare leggendo i risultati di accuracy e val_accuracy che la differenza si sta accentuando, questo significa che il modello si adatta meglio al dataset di training, ma sta perdendo la sua capacità di fare predizioni sui futuri nuovi dati, aumentando la condizione di overfitting.   

Utilizzando softmax:  
In questo caso utilizzando due strati con funzione di attivazione SoftMax inizialmente si puo notare un peggioramento dell'accuratezza ma anche della funzione di perdita.  
D'altra parte l'overfitting è minimo e anche se la percentuale di predeizione è calata si ha pero una convergenza migliore tra i valori del training e del validation set.  
Epoch 10/10  
1875/1875 [==============================] - 9s 5ms/step - loss: 0.2168 - accuracy: 0.9434 - val_loss: 0.2457 - val_accuracy: 0.9359  

Utilizzando softplus:  
Qui invece abbiamo un miglioramento rispetto al modello iniziale già migliorato, l'overfitting è minimo rispetto i valori precedenti e la qualità della predizione sul validation set risulta maggiore.  

Epoch 10/10  
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0214 - accuracy: 0.9933 - val_loss: 0.0721 - val_accuracy: 0.9807  


**Esercizio 3**  

Adatto la rete per l'utilizzo di sparse_categorical_crossentropy:


> #y_train_cat = tf.keras.utils.to_categorical(y_train)
> print(y_train_cat[0])
> #y_test_cat = tf.keras.utils.to_categorical(y_test)
> xin = Input(shape=(784))
> x = Dense(128,activation='relu')(xin)
> res = Dense(10,activation='softmax')(x)

> mynet2 = Model(inputs=xin,outputs=res)
> mynet2.summary()

> mynet2.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
> mynet2.fit(x_train,y_train, shuffle=True, epochs=10, batch_size=32,validation_data=(x_test,y_test))
