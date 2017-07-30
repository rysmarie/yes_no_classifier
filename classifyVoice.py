import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import pyaudio
import wave

import recorder

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

n_datas = 20
X = []
y = []
for i in range(n_datas):
    no, _ = sf.read('no_{}.wav'.format(i))
    yes, _ = sf.read('yes_{}.wav'.format(i))
    no = no[:, 0]
    yes = yes[:, 0] #面倒くさいので片方だけ
    X.append(no)
    X.append(yes)
    y.append(0)
    y.append(1)

X = np.array(X)
y = np.array(y)
X_fft = np.array([np.fft.fft(x) for x in X])
X_fft = np.fft.fft(X)
X = np.array([np.hstack((x.real**2+x.imag**2, np.arctan2(x.real, x.imag))) for x in X_fft])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print('accuracy score:', accuracy_score(y_test, clf.predict(X_test)))

rec = recorder.WaveRecorder()
yesno = ['No', 'Yes']
while True:
    print('Press input to start recording. Type end to finish recording.')
    if input() == 'end':
        break

    rec.record('output.wav')
    wav, _ = sf.read('output.wav')
    wav = np.array(wav[:, 0])
    wf = np.fft.fft(wav)
    wav = np.hstack((wf.real**2+wf.imag**2, np.arctan2(wf.real, wf.imag)))
    pred = clf.predict(np.array([wav]))
    print(yesno[int(pred)])


