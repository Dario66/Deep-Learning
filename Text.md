
Questo elaborato è stato costruito a partire da un implementazine esistente situata al seguente indirizzo:  
https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-fashion-mnist-clothing-classification/?fbclid=IwAR3nCTzPREKJ_D7sDDIlEE9FTWcBAGHIyaFKq3_LJCz-D7jkjVNHTWOQNf8  

Tale esempio è stato riadattato sia a livello di codice che a livello di traduzione ed eseguito sulla piattaforma di Google Colab per determinarne i risultati, che saranno poi visibili in seguito.  


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


Il model ha anche una funzione chiamata summary() che creerà un riepilogo per il modello.

![alt text](https://github.com/Dario66/Deep-Learning/blob/main/img0.png)  

*Total params: 542,230*
*Trainable params: 542,230*
*Non-trainable params: 0*

## Valutazione del modello

Dopo aver definito il modello, dobbiamo valutarlo. Il modello sarà valutato usando la procedura 5-fold cross validation.  
Il valore di k=5 è stato scelto per fornire una linea di base per la valutazione e per non essere troppo grande da richiedere un lungo tempo di esecuzione.   
Ogni testset sarà il 20% del training dataset, vicino alla grandezza dell'attuale testset per questo problema.  
Il training dataset viene mischiato prima di essere diviso e il mischiamento dei campioni viene eseguito ogni volta in modo che qualsiasi modello che valutiamo, avrà gli stessi set di dati di train e di test, fornendo un confronto alla pari.  
Addestreremo il modello di base per un modesto periodo di 10 epoche di addestramento con un batch size predefinito di 32 esempi. Il set di test per ogni fold sarà utilizzato per valutare il modello sia durante ogni epoca dell'addestramento, in modo da poter creare successivamente delle curve di apprendimento, sia alla fine dell'esecuzione, in modo da poter stimare le prestazioni del modello. Come tale, terremo traccia della storia risultante da ogni esecuzione, così come l'accuratezza di classificazione.  

>*# valuta il modello k-fold cross-validation*  
>**def evaluate_model(dataX, dataY, n_folds=5):**     
>  **scores, histories = list(), list()**  
>  *# prepara cross validation*  
>  **kfold = KFold(n_folds, shuffle=True, random_state=1)**   
>  *# enumera le divisioni*   
>  **for train_ix, test_ix in kfold.split(dataX):**  
>    *# definisce il model*   
>    **model = define_model()**  
>    *# seleziona le righe per il train e il test*  
>    **trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]**  
>    *# fit del modello*  
>    **history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)**  
>    *# valuta il modello*  
>    **_, acc = model.evaluate(testX, testY, verbose=0)**  
>    **print('> %.3f' % (acc * 100.0))**  
>    *# appende I risultati*  
>    **scores.append(acc)**  
>    **histories.append(history)**  
>  **return scores, histories**  


**n_folds:** dividere il set di dati di allenamento in k folds  
**shuffle:** per indicare se il dataset dev'essere mischiato.  
Quando shuffle è True, random_state influenza l'ordine degli indici, che controlla la casualità di ogni piega.  
**random_state:** Quando shuffle è True, random_state influenza l'ordine degli indici, che controlla la casualità di ogni fold.  

**fit():** Addestra il modello per un numero fisso di epoche (iterazioni su un dataset).  
**x_train:** uint8 NumPy Array di dati di immagini in scala di grigi con 60000 forme da 28x28 pixel per i dati di training.   
**y_train:** uint8 NumPy array di etichette (interi nell'intervallo 0-9) con 60000 forme per i dati di training.
epochs:Intero. Numero di epoche per addestrare il modello. Un'epoca è un'iterazione su tutti i dati x e y forniti   
**batch_size:** Intero. Numero di campioni per l'aggiornamento del gradiente.  
**validation_data:** Dati su cui valutare la perdita e qualsiasi metrica del modello alla fine di ogni epoca.  
**verbose:** Impostando verbose 0, 1 o 2 si dice come si vuole "vedere" il progresso dell'allenamento per ogni epoca.  
**evaluate():** Restituisce il valore di perdita e i valori di metrica per il modello in modalità test  

## Risultati attuali


Ci sono due aspetti chiave da presentare: la diagnostica del comportamento di apprendimento del modello durante l'addestramento e la stima delle prestazioni del modello. Questi possono essere implementati utilizzando funzioni separate.  
In primo luogo, la diagnostica prevede la creazione di un grafico lineare che mostra le prestazioni del modello sul train e il Test set durante ciascuna fold della k-fold cross-validation. QQuesti grafici sono preziosi per avere un'idea se un modello è in overfitting, underfitting, o ha un buon adattamento al set di dati.  

Creeremo una singola figura con due sottotrame, una per la perdita e una per l'accuratezza. Le linee blu indicheranno le prestazioni del modello nel training dataset e le linee arancioni indicheranno le prestazioni nel dataset di test. La funzione summarize_diagnostics() crea e mostra questo grafico in base alle cronologie di allenamento raccolte.  

>*# traccia curve di apprendimento diagnostiche*  
>**def summarize_diagnostics(histories):**  
>  **for i in range(len(histories)):**  
>    *# traccia la perdita*  
>    **pyplot.subplot(211)**  
>    **pyplot.title('Cross Entropy Loss')**   
>    **pyplot.plot(histories[i].history['loss'], color='blue', label='train')**   
>    **pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')**   
>    *# traccia l'accuratezza*  
>    **pyplot.subplot(212)**  
>    **pyplot.title('Classification Accuracy')**  
>    **pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')**  
>    **pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')**  
>  **pyplot.show()**   


**subplot():** parametri della griglia del subplot codificati come un singolo intero.
**title():** aggiunge un titolo
**plot():** dedicato alla visualizzazione delle linee, Disegna tutte le linee nella stessa trama.
**show():** Visualizza tutte le figure.

I punteggi di accuratezza della classificazione raccolti durante ogni piega possono essere riassunti calcolando la media e la deviazione standard. Ciò fornisce una stima della performance media attesa del modello addestrato su questo set di dati, con una stima della varianza media nella media.  
Riassumeremo anche la distribuzione dei punteggi creando e mostrando un box and whisker plot.
La funzione summary_performance() di seguito lo implementa per un determinato elenco di punteggi raccolti durante la valutazione del modello.  


>*# riassume le prestazioni del modello*  
>**def summarize_performance(scores):**  
>  *# stampa il riepilogo*  
>  **print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))**  
>  *# box and whisker plots dei risultati*  
>  **pyplot.boxplot(scores)**  
>  **pyplot.show()**  

## Esempio completo

>**from matplotlib import pyplot**  *#libreria  per la creazione di visualizzazioni statiche, animate e interattive in Python*  
>**from keras.datasets import fashion_mnist**  *# carica il Fashion-MNIST dataset*  
>**import tensorflow as tf**  
>**from sklearn.model_selection import KFold**  
>**from statistics import mean**  
>**import numpy as np**  
>
>**(trainX, trainY), (testX, testY) = fashion_mnist.load_data()**  
> *# summarize loaded dataset*  
>**print('Train: X=%s, Y=%s' % (trainX.shape, trainY.shape))**  
>**print('Test: X=%s, Y=%s' % (testX.shape, testY.shape))**  
>
> *# carica dataset*  
>**(trainX, trainY), (testX, testY) = fashion_mnist.load_data()**  
> *# rimodella il dataset per avere un singolo canale*  
>**trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))**  
>**testX = testX.reshape((testX.shape[0], 28, 28, 1))**  
>
> *# one hot encode target values*  
>**trainY =tf.keras.utils.to_categorical(trainY)**  
>**testY = tf.keras.utils.to_categorical(testY)**  
>
> *# load train and test dataset*  
>**def load_dataset():**  
>   *# load dataset*  
>  **(trainX, trainY), (testX, testY) = fashion_mnist.load_data()**  
>   *# reshape dataset to have a single channel*  
>  **trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))**  
>  **testX = testX.reshape((testX.shape[0], 28, 28, 1))**  
>   *# one hot encode target values*  
>  **trainY = tf.keras.utils.to_categorical(trainY)**  
>  **testY = tf.keras.utils.to_categorical(testY)**  
>  **return trainX, trainY, testX, testY**  
> 
> *# scale pixels*  
>**def prep_pixels(train, test):**  
>   *# convert from integers to floats*   
>  **train_norm = train.astype('float32')**  
>  **test_norm = test.astype('float32')**  
>   *# normalize to range 0-1*   
>  **train_norm = train_norm / 255.0**  
>  **test_norm = test_norm / 255.0**  
>   *# return normalized images*  
>  **return train_norm, test_norm**  
> 
> *# define cnn model*  
>**def define_model():**  
>  **model = tf.keras.Sequential()**  
>  **model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))**  
>  **model.add(MaxPooling2D((2, 2)))**  
>  **model.add(Flatten())**  
>  **model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))**  
>  **model.add(Dense(10, activation='softmax'))**  
>   *# compile model*  
>  **opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)**  
>  **model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])**  
>  **return model**  
> 
> *# evaluate a model using k-fold cross-validation*  
>**def evaluate_model(dataX, dataY, n_folds=5):**  
>  **scores, histories = list(), list()**  
>   *# prepare cross validation*  
>  **kfold = KFold(n_folds, shuffle=True, random_state=1)**  
>   *# enumerate splits*  
>  **for train_ix, test_ix in kfold.split(dataX):**  
>     *# define model*  
>    **model = define_model()**  
>     *# select rows for train and test*  
>    **trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]**  
>     *# fit model*  
>    **history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)**  
>     *# evaluate model*  
>    **_, acc = model.evaluate(testX, testY, verbose=0)**  
>    **print('> %.3f' % (acc * 100.0))**  
>     *# append scores*   
>    **scores.append(acc)**  
>    **histories.append(history)**  
>  **return scores, histories**  
> 
> *# plot diagnostic learning curves*  
>**def summarize_diagnostics(histories):**  
>  **for i in range(len(histories)):**  
>     *# plot loss*  
>    **pyplot.subplot(211)**  
>    **pyplot.title('Cross Entropy Loss')**  
>    **pyplot.plot(histories[i].history['loss'], color='blue', label='train')**  
>    **pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')**  
>     *# plot accuracy*  
>    **pyplot.subplot(212)**  
>    **pyplot.title('Classification Accuracy')**  
>    **pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')**  
>    **pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')**  
>  **pyplot.show()**  
> 
> *# summarize model performance*  
>**def summarize_performance(scores):**  
>   *# print summary*  
>  **print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, np.std(scores)*100, len(scores)))**  
>   *# box and whisker plots of results*  
>  **pyplot.boxplot(scores)**  
>  **pyplot.show()**  
> 
> *# run the test harness for evaluating a model*  
>**def run_test_harness():**   
>   *# load dataset*  
>  **trainX, trainY, testX, testY = load_dataset()**  
>   *# prepare pixel data*    
>  **trainX, testX = prep_pixels(trainX, testX)**  
>   *# evaluate model*  
>  **scores, histories = evaluate_model(trainX, trainY)**  
>   *# learning curves*  
>  **summarize_diagnostics(histories)**  
>   *# summarize estimated performance*  
>  **summarize_performance(scores)**  
> 
> *# entry point, run the test harness*   
>**run_test_harness()**  

L'esecuzione dell'esempio stampa l'accuratezza di classificazione per ogni fold del processo di cross-validation. Questo è utile per avere un'idea del progresso della valutazione del modello.  

I risultati possono variare a causa della natura stocastica dell'algoritmo o della procedura di valutazione, o delle differenze nella precisione numerica. Considera di eseguire l'esempio alcune volte e confronta il risultato medio.  

Possiamo vedere che per ogni piega, il modello di base ha raggiunto un tasso di errore inferiore al 10%.  

![alt text](https://github.com/Dario66/Deep-Learning/blob/main/img3.png)    


Successivamente, viene mostrato un grafico diagnostico, che dà un'idea del comportamento di apprendimento del modello attraverso ogni fold.  
In questo caso, possiamo vedere che il modello generalmente raggiunge un buon adattamento, con le curve di apprendimento di train e test che convergono. Ci possono essere alcuni segni di leggero overfitting.  

![alt text](https://github.com/Dario66/Deep-Learning/blob/main/img4.png)   

Successivamente, viene calcolato il riepilogo delle prestazioni del modello.    

![alt text](https://github.com/Dario66/Deep-Learning/blob/main/img6.png)    

Infine, viene creato un diagramma per riassumere la distribuzione dei punteggi di accuratezza:  

![alt text](https://github.com/Dario66/Deep-Learning/blob/main/img5.png)   

Ora disponiamo di un solido cablaggio di prova e di un modello di base dalle buone prestazioni.  


## Come sviluppare un modello migliore  

Ci sono molti modi in cui potremmo esplorare miglioramenti al modello di base.  
Esamineremo le aree che spesso si traducono in un miglioramento. Il primo sarà una modifica all'operazione convoluzionale per aggiungere il padding e il secondo si baserà su questo per aumentare il numero di filtri.  

## Padding Convolutions

L'aggiunta di padding all'operazione convoluzionale può spesso comportare migliori prestazioni del modello.  
Per impostazione predefinita, l'operazione convoluzionale utilizza il padding "valid", il che significa che le convoluzioni vengono applicate solo ove possibile. Questo può essere modificato in "same" in modo che i valori 0 vengano aggiunti attorno all'input in modo tale che l'output abbia le stesse dimensioni dell'input.   

