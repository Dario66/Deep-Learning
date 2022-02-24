

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


![alt text](https://github.com/Dario66/Deep-Learning/blob/main/img1.png)  

Viene creato un grafico delle prime nove immagini del DataSet che mostra immagini in scala di grigi di capi di abbigliamento.

Breve descrizione delle funzioni usate:  
**pyplot.subplot(330 + 1 + i):** #parametri della griglia del subplot codificati come un singolo intero.   
Es. "111" significa "griglia 1x1, prima sottotrama" e "234" significa "griglia 2x3, 4a sottotrama". Nel nostrocaso la griglia sarà 3x3 con 9 sottotrame.  
**pyplot.imshow():** #visualizza l’immagine impostando il parametro opzionale cmap per la scala di grigi.

## Sviluppo di un modello di base  
Questo è fondamentale perché comporta sia lo sviluppo dell'infrastruttura in modo che qualsiasi modello che progettiamo possa essere valutato sul set di dati, e stabilisce una linea di base nelle prestazioni del modello sul problema, con cui tutti i miglioramenti possono essere confrontati.  

Il design è modulare, e possiamo sviluppare una funzione separata per ogni pezzo. Questo permette che un dato aspetto possa essere modificato o intercambiato, se lo desideriamo, separatamente dal resto.  
Possiamo svilupparlo con cinque elementi chiave. Sono il caricamento del dataset, la preparazione del dataset, la definizione del modello, la valutazione del modello e la presentazione dei risultati.  

Sul DataSet sappiamo alcune cose, per esempio che le immagini sono tutte pre-segmentate (ad esempio, ogni immagine contiene un singolo capo di abbigliamento), che le immagini hanno tutte la stessa dimensione quadrata di 28×28 pixel, e che le immagini sono in scala di grigi.  Pertanto, possiamo caricare le immagini e rimodellare gli array di dati per avere un singolo canale di colore.  

>*# rimodella il dataset per avere un singolo canale*  
>**trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))**  
>**testX = testX.reshape((testX.shape[0], 28, 28, 1))**  

Sappiamo anche che ci sono 10 classi e che le classi sono rappresentate come interi unici.
Possiamo, quindi, usare una codifica a un solo punto per l'elemento di classe di ogni campione, trasformando l'intero in un vettore binario di 10 elementi con un 1 per l'indice del valore della classe.   Possiamo ottenere questo con la funzione di utilità to_categorical().   

>*# codifica dei valori*   
>**trainY =tf.keras.utils.to_categorical(trainY)**  
>**testY = tf.keras.utils.to_categorical(testY)**  

## Preparare i dati dei pixel  

Sappiamo che i valori dei pixel per ogni immagine nel DataSet sono interi unsigned nel range tra nero e bianco oppure da 0 a 255.  
Non sappiamo il miglior modo per scalare i valori dei pixel per la modellazione, ma sappiamo che sarà necessario un certo ridimensionamento.  

Un buon punto di partenza è normalizzare i valori dei pixel delle immagini in scala di grigi, per esempio ridimensionarli nell'intervallo [0,1].   Questo comporta prima la conversione del tipo di dati da interi senza segno a float, poi dividere i valori dei pixel per il valore massimo.  

>**def prep_pixels(train, test):**  
>*# converte da intero a float*  
> **train_norm = train.astype('float32')**    
> **test_norm = test.astype('float32')**    
> *# normalizzazzione (0-1)*  
> **train_norm = train_norm / 255.0**  
> **test_norm = test_norm / 255.0**  
>  *# ritorna le immagini normalizzate*  
> **return train_norm, test_norm**  

La funzione prep_pixels()implementa questi comportamenti e viene fornita con i valori dei pixel per entrambi i set di dati di train e test che dovranno essere scalati.  
Questa funzione deve essere chiamata per preparare i valori dei pixel prima di qualsiasi modellazione.  

## Definire il modello  
Obbiettivo è definire un modello di rete neurale convoluzionale di base per il problema.  
Il modello ha due aspetti principali:  
L’estrazione delle feature, comprendente strati di convoluzione e pooling e il classificatore, incaricato di fare le previsioni.  

Per lo strato di convoluzione, possiamo iniziare con un singolo strato convoluzionario con un filtro di piccole dimensioni (3,3) e un numero modesto di filtri (32) seguito da uno strato di pooling massimo.  Le mappe dei filtri possono poi essere appiattite per fornire caratteristiche al classificatore.  

Dato che il problema è una classificazione multiclasse, sappiamo che avremo bisogno di uno strato di output con 10 nodi per prevedere la distribuzione di probabilità di un'immagine appartenente a ciascuna delle 10 classi. Questo richiederà anche l'uso di una funzione di attivazione softmax. Tra l'estrattore di caratteristiche e lo strato di output, possiamo aggiungere uno strato per interpretare le caratteristiche, in questo caso con 100 nodi.

Tutti gli strati useranno la funzione di attivazione ReLU e lo schema di inizializzazione dei pesi He.

Useremo una configurazione conservativa per l’ottimizzatore “stochastic gradient descent”, con un learning rate di 0.01 e un momentum di 0.9.  
La funzione di perdita cross-entropy sarà ottimizzata, dal momento che è adatta alla classificazione multiclasse e sara controllata la metrica di accuratezza della classificazione, che è appropriata dato che abbiamo lo stesso numero di esempi in ciascuna delle 10 classi.  

Creiamo una funzione python per definire e restituire questo modello.  

>  *# definisce il modello*  
>**def define_model():**  
>  **model = tf.keras.Sequential()**  
>  **model.add(tf.keras.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))**  
>  **model.add(tf.keras.MaxPooling2D((2, 2)))**  
>  **model.add(tf.keras.Flatten())**  
>  **model.add(tf.keras.Dense(100, activation='relu', kernel_initializer='he_uniform'))**  
>  **model.add(tf.keras.Dense(10, activation='softmax'))**  
>  *# compile model*  
>  **opt = tf.keras.SGD(lr=0.01, momentum=0.9)**  
>  **model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])**  
>  **return model**  

**tf.keras.Sequential():** Un modello sequenziale è il tipo più semplice di modello, una pila lineare di livelli. Ma ci sono alcuni difetti nell'uso dell'API del modello sequenziale, è limitato in certi punti. Non possiamo costruire reti complesse usando questa API ma la utilizzeremo solo per l’esempio.  
**model.add():** metodo per creare un modello sequenziale in modo incrementale.  
**Conv2D():** definisce un layer convoluzionale, si devono specificare :  

**•	filters:** Intero, la dimensionalità dello spazio di uscita (cioè il numero di filtri di uscita nella convoluzione).  
**•	kernel_size:** specifica l'altezza e la larghezza della finestra di convoluzione 2D.  
**•	activation:** Funzione di attivazione da utilizzare. Se non si specifica nulla, non viene applicata alcuna attivazione  
**•	kernel_initializer:** nizializzatore per la matrice dei pesi del kernel, definiscono il modo di impostare i pesi casuali iniziali dei livelli di Keras. Vi sono molte classi disponibili in questo esempio utilizziamo “he_uniform”  
**•	Input shape:** Le forme sono tuple che rappresentano quanti elementi ha una matrice o un tensore in ciascuna dimensione.  
**MaxPooling2D:** Aggiunge un livello di MaxPooling2D, il compito di questo livello è ridurre la complessità del modello ed estrarre le caratteristiche locali trovando i valori massimi per ogni pool 2 x 2, il primo parametro è pool_size, specificando un solo intero la stessa lunghezza della finestra sarà usata per entrambe le dimensioni(2x2), mentre il secondo, strides, specifica quanto lontano si sposta la finestra di pooling per ogni passo.  
**Flatten():** aggiunge un livello e appiattisce i dati di input, la forma di output deve utilizzare tutti i parametri esistenti concatenandoli.  
**Dense():** strato in cui tutti i nodi dello strato precedente sono connessi a tutti i nodi dello strato successivo tramite dei pesi. Come parametri accetta:  
**•	units:** Intero positivo, dimensionalità dello spazio di output  
**•	activation:** Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).  
**•	kernel_initializer:** Inizializzatore per la matrice dei pesi del kernel.  
**•	activation ():** Funzione di attivazione da utilizzare.  
**SGD:** è un metodo iterativo per l'ottimizzazione di funzioni differenziabili, approssimazione stocastica del metodo di gradient descent.  
**•	learning_rate:** Il tasso di apprendimento  
**•	momentum:** iperparametro di tipo float >= 0  
**Compile():** Configura il modello per l'addestramento:  
**•	optimizer:** Stringa, (nome dell'ottimizzatore) o istanza dell'ottimizzatore  
**•	loss():** Funzione di perdita  
**•	metrics():** Elenco delle metriche che devono essere valutate dal modello durante l'addestramento e il test  




