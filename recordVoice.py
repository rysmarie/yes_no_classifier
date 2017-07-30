import sys
import pyaudio
import wave

import recorder


cnt = 0
print('input prefix of output filename:')
fname = input()
print('input number to start with:')
cnt = int(input())
rec = recorder.WaveRecorder()
while True:
    print('Press enter to start recoding. Type end to finish recording.')
    if input() == 'end':
        break
    rec.record('{}_{}.wav'.format(fname, cnt))
    cnt += 1
