import numpy as np
import matplotlib.pyplot as plt

tp = 0.75
t0 = 5
N = 100
ExcitationFunction = []
time = np.linspace(start=0, stop=10, num=N)
ts = (time[1]-time[0])
fs = 1/ts

for dtime in time:
    ExcitationFunction_tmp = (-(dtime-t0)/tp)*np.exp(0.5-np.power(dtime-t0, 2)/(2*np.power(tp, 2)))
    ExcitationFunction.append(ExcitationFunction_tmp)

plt.figure(1)
plt.plot(time, ExcitationFunction)
plt.title('Time Domain')
plt.xlabel('time/ns')

y = np.fft.fft(ExcitationFunction)
freq = np.fft.fftfreq(N, ts)

plt.figure(2)
plt.plot(freq[0: int(N/2)], abs(y[0: int(N/2)]))
plt.title('Frequency Domain')
plt.ylabel('Amplitude')
plt.xlabel('Frequency/GHz')
plt.show()

# exc File save
exc = np.zeros((N, 2))
exc[:, 0] = np.array(time)
exc[:, 1] = np.array(ExcitationFunction)

np.savetxt('exc_{}.exc'.format(N), exc)

print('--end--')
