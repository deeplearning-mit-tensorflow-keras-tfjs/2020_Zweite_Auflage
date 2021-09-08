## 🖥Quellcode

Auf dieser Seite finden Sie den Quellcode zum Buch - 2. Auflage

## Kapitel 3: Neuronale Netze

Abschnitt | Kurzbeschreibung | Dateipfad | Hinweise
------|------------------|-----------|---------
3.4 |Klassifikation von Iris-Blüten mit scikit learn | [chap\_3/iris\_classification.py](chap\_3/iris\_classification.py) [chap\_3/iris\_classification.ipynb](chap\_3/iris\_classification.ipynb) | Dataset: [iris.csv](chap\_3/iris.csv)

## Kapitel 4: Python und Machine-Learning-Bibliotheken

Abschnitt | Kurzbeschreibung | Dateipfad | Hinweise
------|------------------|-----------|---------
4.5.1 |Beispiele von Funktionalitäten von NumPy | [chap\_4/numpy\_examples.py](chap\_4/numpy\_examples.py) [chap\_4/numpy\_examples.ipynb](chap\_4/numpy\_examples.ipynb)| Installieren Sie die zwei Python-Packages </br> `pip install tabulate wget`. 
4.6.4 |Visualisierung vom Olivetti-Dataset|[chap\_4/olivetti\_dataset.py](chap\_4/olivetti\_dataset.py) [chap\_4/olivetti\_dataset.ipynb](chap\_4/olivetti\_dataset.ipynb) 
4.6.5 |Normalisierung von Daten mit scikit-learn|[chap\_4/normalize\_iris\_dataset.py](chap\_4/normalize\_iris\_dataset.py) [chap\_4/normalize\_iris\_dataset.ipynb](chap\_4/normalize\_iris\_dataset.ipynb)
4.6.6 |Benutzung von Seed |[chap\_4/seed_example.py](chap\_4/seed_example.py) [chap\_4/seed_example.ipynb](chap\_4/seed_example.ipynb)
4.7 |Lineares Regressionsmodell mit scikit-learn|[chap\_4/linear\_regression.py](chap\_4/linear\_regression.py) [chap\_4/linear\_regression.ipynb](chap\_4/linear\_regression.ipynb)

## Kapitel 5: TensorFlow

☝ Tipp: Möchten Sie weniger Debugging-Ausgaben von TensorFlow erhalten, können Sie den Debug-Level mit folgenden Python-Zeilen verändern:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

Abschnitt | Kurzbeschreibung | Dateipfad | Hinweise
------|------------------|-----------|---------
5.2.7 |Hello World in TensorFlow|[chap\_5/hello\_world.py](chap\_5/hello\_world.py) [chap\_5/hello\_world.ipynb](chap\_5/hello\_world.ipynb) |
5.4 |Beispiele mit Tensoren|[chap\_5/tensors\_dimensions.py](chap\_5/tensors\_dimensions.py) [chap\_5/tensors\_dimensions.ipynb](chap\_5/tensors\_dimensions.ipynb)
5.4 |Bild in Tensoren laden|[chap\_5/loading\_picture\_in\_tensors.py](chap\_5/loading\_picture\_in\_tensors.py) [chap\_5/loading\_picture\_in\_tensors.ipynb](chap\_5/loading\_picture\_in\_tensors.ipynb)
5.4.1 |Beispiel der Verwendung von tf.Variable() und tf.assign()| [chap\_5/variables.py](chap\_5/variables.py) [chap\_5/variables.ipynb](chap\_5/variables.ipynb)
5.5.1 |Konzept von einem Graph mit TensorFlow|[chap\_5/simple_graph.py](chap\_5/simple_graph.py) [chap\_5/simple_graph.ipynb](chap\_5/simple_graph.ipynb)
5.5.4 |Benutzung von AutoGraph mit @tf.function|[chap\_5/simple_graph_with_autograph.py](chap\_5/simple_graph_with_autograph.py) [chap\_5/simple_graph_with_autograph.ipynb](chap\_5/simple_graph_with_autograph.ipynb)
5.6 |Benutzung der CPU und GPU| [chap\_5/tensors_gpu.py](chap\_5/tensors_gpu.py) [chap\_5/tensors_gpu.ipynb](chap\_5/tensors_gpu.ipynb)
5.7.3 |Projekt: Eine lineare Regression| [chap\_5/linear\_regression\_model.py](chap\_5/linear\_regression\_model.py) [chap\_5/linear\_regression\_model.ipynb](chap\_5/linear\_regression\_model.ipynb)|Python-Skript korrigiert: *input* mit *x* und *output* mit y ersetzen. <br/> *(Dank an den aufmerksamen Leser Michael S.!)*
5.8.2 |Automatische Konvertierung mit tf_upgrade_v2| `tf_upgrade_v2 --intree tf1 --outtree tf2`

## Kapitel 6: Keras

Abschnitt | Kurzbeschreibung | Dateipfad | Hinweise
------|------------------|-----------|---------
6.1.1 |Version von Keras überprüfen|[chap\_6/keras\_version\_check.py](chap\_6/keras\_version\_check.py) [chap\_6/keras\_version\_check.ipynb](chap\_6/keras\_version\_check.ipynb)
6.2.1 |Benutzung der Sequential API|[chap\_6/keras\_xor\_sequential.py](chap\_6/keras\_xor\_sequential.py) [chap\_6/keras\_xor\_sequential.ipynb](chap\_6/keras\_xor\_sequential.ipynb)|
6.2.2 |Benutzung der Functional API|[chap\_6/keras\_xor\_functional.py](chap\_6/keras\_xor\_functional.py) [chap\_6/keras\_xor\_functional.ipynb](chap\_6/keras\_xor\_functional.ipynb)|
6.5 |Laden und speichern von Modellen| [chap\_6/keras\_load\_save.py](chap\_6/keras\_load\_save.py) [chap\_6/keras\_load\_save.ipynb](chap\_6/keras\_load\_save.ipynb)|Um die Zeile ```tfjs.converters.save_keras_model(addition_model, "./addition_model")``` benutzen zu können, muss tfjs installiert werden, damit das Modell auch mit TensorFlow.js exportiert werden kann. `pip install tensorflowjs` und im Python-Code `import tensorflowjs as tfjs`
6.6 |Benutzung von Keras Applications|[chap\_6/keras\_applications\_list.py](chap\_6/keras\_applications\_list.py) [chap\_6/keras\_applications\_list.ipynb](chap\_6/keras\_applications\_list.ipynb)
6.8 |Projekt 1: Iris-Klassifikation mit Keras (ohne Evaluationsmetriken)|[chap\_6/keras\_iris\_classification.py](chap\_6/keras\_iris\_classification.py) [chap\_6/keras\_iris\_classification.ipynb](chap\_6/keras\_iris\_classification.ipynb)|Dataset: [iris.csv](chap\_6/data/iris.csv)
6.8 |Iris-Klassifikation mit Keras (mit Evaluationsmetriken)|[chap\_6/keras\_iris\_classification\_with\_evaluation.py](chap\_6/keras\_iris\_classification\_with\_evaluation.py) [chap\_6/keras\_iris\_classification\_with\_evaluation.ipynb](chap\_6/keras\_iris\_classification\_with\_evaluation.ipynb)|Dataset: [iris.csv](chap\_6/data/iris.csv) <br> Python-Skript korrigiert: Ausgabereihenfolge angepasst.<br> *(Dank an den aufmerksamen Leser Elias!)*
6.9 |Projekt 2: CNNs mit Fashion-MNIST|[chap\_6/keras\_fashion\_cnn.py](chap\_6/keras\_fashion\_cnn.py) [chap\_6/keras\_fashion\_cnn.ipynb](chap\_6/keras\_fashion\_cnn.ipynb) |
6.10 |Projekt 3: Ein einfaches CNN mit dem CIFAR-10-Dataset|[chap\_6/keras\_cnn\_cifar\_test.py](chap\_6/keras\_cnn\_cifar\_test.py) [chap\_6/keras\_cnn\_cifar\_test.ipynb](chap\_6/keras\_cnn\_cifar\_test.ipynb)
6.11 |Aktienkursvorhersage|[chap\_6/keras\_stock\_prediction.py](chap\_6/keras\_stock\_prediction.py)|Dataset: [tsla.csv](chap\_6/data/tsla.csv)

## Kapitel 7: Netze und Metriken visualisieren

Abschnitt | Kurzbeschreibung | Dateipfad | Hinweise
------|------------------|-----------|---------
7.1.1 |Graphen visualisieren/Histogram,  Distributions Dashboard|[chap\_7/tensorboard/keras_fashion_cnn.py](chap\_7/tensorboard/keras_fashion_cnn.py)|Bitte die vier im Code erwähnten Datasets (*.gz-Dateien) von https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion herunterladen und unter data/fashion/ speichern. <br/> *(Dank an einen aufmerksamen Leser/eine aufmerksame Leserin!)*
7.1.4 |TensorBoard: Benutzung des Text-Dashboard|[chap\_7/tensorboard/text\_summary\_reuters.py](chap\_7/tensorboard/text\_summary\_reuters.py) 
7.1.4 |with tf.name_scope|[chap\_7/tensorboard/text\_summary\_scope.py](chap\_7/tensorboard/text\_summary\_scope.py) 
7.1.5 |Image Summary|[chap\_7/tensorboard/image_summary.py](chap\_7/tensorboard/image_summary.py)
7.1.6 |Integration TensorBoard in Jupyter | [chap\_7/jupyter/tensorboard\_integration.ipynb](chap\_7/jupyter/tensorboard\_integration.ipynb)
7.3.1 |TF 1.x - Debugging mit TensorBoard | [chap\_7/tensorboard/tf1/simple\_net\_graph.py](chap\_7/tensorboard/tf1/simple\_net\_graph.py)|TensorBoard starten:</br>`tensorboard --logdir="./logs" --debugger_port=12345`
7.3.2 |TF 1.x - Debugging eines CNNs | [chap\_7/tensorboard/tf1/tensorboard/debugger\_example.py](chap\_7/tensorboard/tf1/tensorboard/debugger\_example.py)
7.3.4 |TF 1.x - Debugging eines CNNs (mit Keras) | [chap\_7/tensorboard/tf1/tensorboard/keras\_debugger\_example.py](chap\_7/tensorboard/tf1/tensorboard/keras_debugger_example.py)
7.5.1 |Keras: Benutzung von plot\_model()|[chap\_7/plot\_model/plot\_model\_example.py](chap\_7/plot\_model/plot\_model\_example.py)|
7.5.2 |Aktivierungen visualisieren|[chap\_7/activations/activations\_vis.py](chap\_7/activations/activations\_vis.py) [chap\_7/activations/activations\_vis.ipynb](chap\_7/activations/activations\_vis.ipynb)|
7.5.3 |tf-explain|[chap\_7/tf\_explain/tf\_explain\_grad\_cam.py](chap\_7/tf\_explain/tf\_explain\_grad\_cam.py)
7.5.4 |Keras-Metriken mit Bokeh darstellen|[chap\_7/bokeh/keras\_history\_bokeh.py](chap\_7/bokeh/keras\_history\_bokeh.py)|Installieren Sie das Python Package *bokeh* </br> `pip install bokeh` 
7.6 |Visualisierung von CNNs mit Quiver|[chap\_7/quiver/quiver\_vgg16.py](chap\_7/quiver/quiver\_vgg16.py)|Bitte legen Sie einen leeren Ordner namens *tmp* im aktuellen Verzeichnis an | Vergessen Sie nicht die notwendigen Anpassungen bei dem Quellcode von Quiver durchzuführen!
7.7 |Projekt KeiVi|[chap\_7/keivi/](chap\_7/keivi/)|Installieren Sie die *node\_modules* für das Projet mit `npm install`
7.8.3 |Benutzung von ConX|[chap\_7/conx/VGG\_19\_with\_ConX.ipynb](chap\_7/conx/VGG\_19\_with\_ConX.ipynb)|Jupyter Notebook starten <br>`jupyter notebook` 

## Kapitel 8: TensorFlow.js

☝ Tipp: Um alle Beispiel zu testen, starten Sie bitte BrowserSync vom Ordner */chap8* ```browser-sync start --server --files "*.*"``` 
Die Beispiele können dann über [http://localhost:3000](http://localhost:3000) aufgerufen werden.

Abschnitt | Kurzbeschreibung | Dateipfad | Hinweise
------|------------------|-----------|---------
8.4 |Operationen mit Tensoren|[chap\_8/examples.html](chap\_8/examples.html)
8.5 |Quadratische Regression (mit tfjs-vis)|[chap\_8/polynomial\_regression](chap\_8/polynomial\_regression)
8.6 (Bonus)|XOR-Modell mit TensorFlow.js|[chap\_8/xor.html](chap\_8/xor.html)
8.7 |PoseNet-Modell|[chap\_8/posenet](chap\_8/posenet)|Bitte passendes MP4-Video in den Ordner *./video* platzieren
8.8 |Intelligente Smart-Home-Komponente mit TensorFlow.js und Node.js|[chap\_8/occupancy](chap\_8/occupancy/)|Dataset: [datatraining.txt](chap\_8\occupancy\data\datatraining.txt)
8.9 |Bildklassifikation mit ml5.js und MobileNet|[chap\_8/ml5\_js](chap\_8/ml5\_js)

## Kapitel 9: Praxisbeispiele

Abschnitt | Kurzbeschreibung | Dateipfad | Hinweise
------|------------------|-----------|---------
9.1 |Projekt 1: Erkennung von Verkehrszeichen mit Keras|[chap\_9/1\_road\_signs\_keras](chap\_9/1\_road\_signs\_keras)|Bilder müssen in den Ordner *./img* platziert werden. Installieren Sie die das Python-Package webcolors `pip install webcolors`
9.1 |Projekt 1: Erkennung von Verkehrszeichen mit Keras|[chap\_9/1\_road\_signs](chap\_9/1\_road\_signs)|Änderung in der Funktion load_roadsigns_data in der Zeile 65: "./img/train" --> rootpath
9.1 |Projekt 1: Erkennung von Verkehrszeichen mit Keras|[chap\_9/1\_road\_signs\_with\_datagenerator](chap\_9/1\_road\_signs\_with\_datagenerator)|Änderung in der Funktion load_roadsigns_data in der Zeile 65: "./img/train" --> rootpath
9.2 |Projekt 2: Intelligente Spurerkennung mit Keras|[chap\_9/2\_lane\_detection\_keras](chap\_9/2\_lane\_detection\_keras)|
9.2.1 |Beispielcode zur Extraktion von einzelnen Frames mit OpenCV|[chap\_9/2\_lane\_detection\_keras/extract\_video\_frames.py](chap\_9/2\_lane\_detection\_keras/extract\_video\_frames.py)| Installieren Sie die das Python-Package cv2 `pip install opencv-python==3.4.2.16` und falls notwendig SciPy 1.1: `pip install scipy==1.1.0` NB: Es gibt einen TensorFlow Bug (in den Versionen 2.0.0 bis 2.0.2), der dazu führt, dass das bereitgestellte kitti-road.h5 Modell nicht korrekt geladen wird. Verwenden Sie daher bitte die Tensorflow-Versionen ab 2.1.<br/>Sie können auch die neuesten Versionen von TensorFlow (>2.2) und von Scipy, OpenCV benutzen, indem Sie die Zeile<br/> ```mask = scipy.misc.toimage(mask, mode="RGBA")``` <br/>durch folgende ersetzen<br/> ```mask = Image.fromarray(mask.astype(np.uint8), mode=None)``` 
9.2.1 |Visualisierung der Labels für das Dataset von Michael Virgo|[chap\_9/2\_lane\_detection\_keras/labels\_viewer.py](chap\_9/2\_lane\_detection\_keras/labels\_viewer.py)
9.2.2 |Code für die Spurerkennung (Modell 1)|[chap\_9/2\_lane\_detection\_keras/lane\_detection.py](chap\_9/2\_lane\_detection\_keras/lane\_detection.py)
9.2.2 |Code für die Spurerkennung (Modell 2) - basierend auf KITTI-Road Dataset|[chap\_9/2\_lane\_detection\_keras/train\_kitti\_road.py](chap\_9/2\_lane\_detection\_keras/train\_kitti\_road.py)|Das Training sollte wegen längeren Berechnungszeiten auf einem Rechner mit GPU durchgeführt werden. Sollte das nicht gegeben sein, kann das trainierte Modell [chap\_9/2\_lane\_detection\_keras/kitti\_road\_model.h5](chap\_9/2\_lane\_detection\_keras/kitti\_road\_model.h5) benutzt werden. </br></br>Im Vorfeld die Bilddateien von [http://www.cvlibs.net/download.php?file=data\_road.zip](http://www.cvlibs.net/download.php?file=data\_road.zip) herunterladen und nach Entzippen diese in den Ordner *./data* platzieren. </br></br>Adaptierter Code von: [https://github.com/6ixNugget/Multinet-Road-Segmentation]. </br></br>Bitte Dimensionen des Videobreiches anpassen, damit das das Dashcam Video kein Teil des Armaturenbretts beinhaltet (siehe Zeile 42)
9.2.2 |Visualisierung (Modell 2)|[chap\_9/2\_lane\_detection\_keras/kitti\_road.py](chap\_9/2\_lane\_detection\_keras/kitti\_road.py)|Sowohl das fertige trainierte Modell [chap\_9/2\_lane\_detection\_keras/kitti\_road\_model.h5](chap\_9/2\_lane\_detection\_keras/kitti\_road\_model.h5) als auch die Videodatei *dash_cam.mp4* befinden sich in der ZIP-Datei von [https://www.rheinwerk-verlag.de/deep-learning-mit-tensorflow-keras-und-tensorflowjs_5040/](https://www.rheinwerk-verlag.de/deep-learning-mit-tensorflow-keras-und-tensorflowjs_5040/)
9.3 |Projekt 3: YOLO und ml5.js|[chap\_9/3\_object\_detection\_yolo\_tfjs](chap\_9/3\_object\_detection\_yolo\_tfjs)|Bitte *index.html* anpassen, damit die passende MP4-Video Datei abgespielt wird
9.4 |Projekt 4: VGG-19 mit Keras benutzen|[chap\_9/4\_vgg\_19\_keras](chap\_9/4\_vgg\_19\_keras)|Platzieren Sie Ihre Bilder in den Ordner *./samples*
9.5 |Projekt 5: Buchstaben- und Ziffernerkennung mit dem Chars74K-Dataset|[chap\_9/5_chars74k](chap\_9/5_chars74k)
9.6 |Projekt 6: Stimmungsanalyse mit Keras|[chap\_9/6\_sentiment\_keras/sentiment.py](chap\_9/6\_sentiment\_keras/sentiment.py)|Installieren Sie den TensorFlow.js Konverter:`pip install tensorflowjs`
9.7 |Projekt 7: Stimmungsanalyse mit TensorFlow.js|[chap\_9/7\_sentiment\_tfjs/](chap\_9/7\_sentiment\_tfjs)|Bitte den generierten Ordner */tfjs\_sentiment\_model* von 6\_sentiment\_keras kopieren, um das TensorFlow.js Modell benutzen zu können. 
9.8 |Projekt 8: Stimmungsanalyse mit TensorFlow.js|[chap\_9/8\_tf_hub](chap\_9/8\_tf_hub)
9.9 |Projekt 9: Hyperparameter-Tuning mit TensorBoard|[chap\_9/9\_hparams](chap\_9/9\_hparams)
9.10 |Projekt 10: (Nur TF1.x) Fashion-MNIST mit TensorFlow-Estimators|[chap\_9/10\_fashion\_mnist\_estimators\_tf1](chap\_9/10\_fashion\_mnist\_estimators\_tf1)|Platzieren Sie Ihre Bilder in den Ordner *./samples* 

## Kapitel 10: Ausblick
Abschnitt | Kurzbeschreibung | Dateipfad | Hinweise
------|------------------|-----------|---------
10.5.1 |AutoKeras|[chap\_10/autokeras/auto\_keras.py](chap\_10/autokeras/auto\_keras.py)|
10.5.2 |Uber Ludwig|[chap\_10/ludwig/model.yaml](chap\_10/ludwig/model.yaml)|Dataset: [chap\_10/ludwig/reuters-allcats.csv](chap\_10/ludwig/reuters-allcats.csv)
