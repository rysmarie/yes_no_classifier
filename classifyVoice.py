import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import pyaudio
import wave

import recorder

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
n_datas = 20
X = []
y = []
for i in range(n_datas):
    no, _ = sf.read('no_{}.wav'.format(i))
    yes, _ = sf.read('yes_{}.wav'.format(i))
    no = no[:, 0]
    yes = yes[:, 0]  # 面倒くさいので片方だけ
    X.append(no)
    X.append(yes)
    y.append(0)
    y.append(1)

X = np.array(X)
y = np.array(y)
X_fft = np.array([np.fft.fft(x) for x in X])
X_fft = np.fft.fft(X)
X = np.array(
    [np.hstack((x.real**2 + x.imag**2, np.arctan2(x.real, x.imag))) for x in X_fft])
clf = RandomForestClassifier()
scores = cross_val_score(clf, X, y, cv=5)
print('score:{:.3f} (+/-{:.3f})'.format(scores.mean(), scores.std()*2))

#Use all data to train model
clf.fit(X, y)

rec = recorder.WaveRecorder()
yesno = ['No', 'Yes']
while True:
    print('Press enter to start recording. Type end to finish recording.')
    if input() == 'end':
        break

    rec.record('output.wav')
    wav, _ = sf.read('output.wav')
    wav = np.array(wav[:, 0])
    wf = np.fft.fft(wav)
    wav = np.hstack((wf.real**2 + wf.imag**2, np.arctan2(wf.real, wf.imag)))
    pred = clf.predict(np.array([wav]))
    print(yesno[int(pred)])
