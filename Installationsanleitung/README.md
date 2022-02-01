## 📝Installationsanleitung

Zum schnellen Einstieg empfehlen wir die Python-Installation mit Anaconda: https://www.anaconda.com/distribution/

NB: Unter Windows müssen Sie, nachdem Anaconda etwa unter C:\Users\<IHRNAME>\Anaconda3 installiert wurde, C:\Users\<IHRNAME>\Anaconda3\condabin zu Ihrer Umgebungsvariable PATH hinzufügen. <br>

### Organisation der Arbeitsumgebung
Wir empfehlen Ihnen ein Projektverzeichnis in Ihrem HOME-Verzeichnis anzulegen, etwa <i>deeplearning_buch</i>:<br>
```mkdir deeplearning_buch```

Dann wechseln Sie zu diesem Verzeichnis und können dort die Beispiele vom Buch speichern.

### Python-Installation mit Anaconda
Erzeugen Sie eine Umgebung namens <i>dl_env</i> mit der Python Version 3.6:<br>
```conda create -n dl_env python=3.6```


Nachdem die Umgebung erzeugt wurde, muss sie nun aktiviert werden:<br>
```source activate dl_env```

Um alle Beispiele der ersten Kapitel des Buches zu bearbeiten, empfehlen wir Ihnen in einer einzigen Aktion folgende Pakete zu installieren:

```conda install numpy scipy pandas scikit-learn matplotlib```<br>
```conda install tensorflow```<br>
```conda install keras```<br>
```conda install tensorflowjs```<br>

Ein Vorteil von conda ist es, dass beim Installieren eines Pakets nicht nur dieses sondern auch alle vom ihm benötigten Pakete mitinstalliert werden.

Wenn Sie die Liste alle definierten Umgebungen bekommen möchten, können Sie folgendes Kommando ausführen:<br>
```conda env list```

#### Starten der Umgebung
Jedesmal, wenn Sie mit dem Buch arbeiten möchten, empfiehlt es sich zum Ihrer Arbeitsverzeichnis zu wechseln und die Python-Umgebung <i>dl_env</i> im Terminal zu aktivieren mit:<br>
```source activate dl_env```

#### Beendigung der Arbeiten 
Nach Beendigung Ihrer Arbeiten sollten Sie die <i>dl_env</i> Umgebung deaktivieren:<br>
```source deactivate dl_env```

### NB: Python-Installation auf M1
Sollten Sei einen M1 Prozessor einsetzen, verweisen wir für die Installtion auf folgende Seiten:<br>
https://betterdatascience.com/install-tensorflow-2-7-on-macbook-pro-m1-pro/<br>
https://www.examplefiles.net/cs/620349<br>
https://towardsdatascience.com/accelerated-tensorflow-model-training-on-intel-mac-gpus-aa6ee691f894<br>
https://naturale0.github.io/2021/01/29/setting-up-m1-mac-for-both-tensorflow-and-pytorch<br>

Im wesentlichen können Sie wie folgt vorgehen:<br>
```conda install nomkl bzw. pip install nomkl```<br>
```conda install -c apple tensorflow-deps -y```<br>
```python -m pip install tensorflow-macos```<br>
```pip install tensorflow-metal```
