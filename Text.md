

L'esempio proposto utilizza il dataset Fashion-MNIST con immagini di articoli di Zalando in formato 28x28 grayscale di 65000 prodotti suddivise in 10 categorie(6500 immagini per categoria).
Il training set ha 55000 immagini, il Test set da 10000.
Il Fashion-MNIST è simile al MNIST DataSet, il quale è utilizzato per classificare le cifre scitte a mano, nel senso che il formato delle immagini, la divisione tra training e test sono simili.


## Classificazione dell'abbigliamento

La mappatura di tutte le 9 classi intero/etichetta è definito in questo modo:  
0: T-shirt/top  
1: Pantaloni  
2: Pullover  
3: vestito  
4: Cappotto  
5: Sandalo  
6: Maglia  
7: Scarpa da ginnastica  
8: Borsa  
9: Stivaletto 

Carichiamo il dataset Fashion-MNIST tramite le API Keras 


> **from matplotlib import pyplot** *#libreria  per la creazione di visualizzazioni statiche, animate e interattive in Python*  
>**from keras.datasets import fashion_mnist** *#carica il Fashion-MNIST dataset*  
>**(trainX, trainy), (testX, testy) = fashion_mnist.load_data()**  
>  
>**print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))**  
>**print('Test: X=%s, y=%s' % (testX.shape, testy.shape))**  


La funzione load_data() ritorna una tupla di array NumPy:

**x_train**: uint8 NumPy Array di dati di immagini in scala di grigi con 60000 forme da 28x28 pixel per i dati di training.  
**y_train**: uint8 NumPy array di etichette (interi nell'intervallo 0-9) con 60000 forme per i dati di training.  
**x_test**: : uint8 NumPy array di dati di immagini in scala di grigi con 10000 forme da 28x28 pixel per i dati di Test.   
**y_test**: uint8 NumPy array di etichette (interi nell'intervallo 0-9) con 10000 forme per i dati di test  


Il risultato eseguito con Colab è il seguente: 

**Train: X=(60000, 28, 28), y=(60000,)**  
**Test: X=(10000, 28, 28), y=(10000,)**

Abbiamo 60000 immagini di 28x28 pixel per il training set e 10000 immagini di 28x28 pixel per il test set.  

Ora creiamo un grafico delle prime 9 immagini nel training set.  

> **for i in range(9):**  
>   **pyplot.subplot(330 + 1 + i)** *#definisce le griglie*   
>   **pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))** *#visualizza l’immagine in scala di grigi*   
> **pyplot.show()** *#visualizza tutte le figure**   




