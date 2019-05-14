import numpy as np
from numpy import exp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from scipy.signal import argrelmax

def rhs(y, t, p):
    v, b, g, a, h, m, n, q = y
    I = p
    dv = -0.73273 * v - 73.059 * n ** 2 * q ** 2 * (v + 97.569) - 54.642 * h * m * (v - 23.526) - 49.969 + I
    db = -1.0 * (1.1718 * exp(-0.066571 * v - 6.9104) + 1.1718 * exp(0.10582 * v + 10.985)) * (
            b - 1.0 / (exp(-0.097432 * v - 6.993) + 1.0))
    dg = -1.0 * (0.1799 * exp(-0.063317 * v - 5.2488) + 0.1799 * exp(0.064089 * v + 5.3128)) * (
            g - 1.0 / (exp(0.059489 * v + 6.4261) + 1.0))
    da = 0.10046 / (exp(-0.069246 * v - 6.5213) + 1.0) - 0.10046 * a
    dh = -1.0 * (0.17914 * exp(-0.10011 * v - 7.3792) + 0.17914 * exp(0.1085 * v + 7.9979)) * (
            h - 1.0 / (exp(0.09422 * v + 7.5018) + 1.0))
    dm = -1.0 * (0.75724 * exp(-0.08455 * v - 6.6466) + 0.75724 * exp(0.097949 * v + 7.6999)) * (
            m - 1.0 / (exp(-0.091197 * v - 5.3235) + 1.0))
    dn = -1.0 * (0.17962 * exp(-0.032534 * v - 1.766) + 0.17962 * exp(0.11223 * v + 6.0918)) * (
            n - 1.0 / (exp(-0.068647 * v - 3.8117) + 1.0))
    dq = -1.0 * (0.96267 * exp(0.06748 * v + 2.5905) + 0.96267 * exp(-0.039659 * v - 1.5225)) * (
            q - 1.0 / (exp(0.10562 * v + 4.8277) + 1.0))

    return [dv, db, dg, da, dh, dm, dn, dq]


y0 = [0, 0, 0, 0, 0, 0, 0, 0]
t = np.arange(0, 100, 0.01) # t0, t1, step
peakTimes = []

freqs = []
currents = range(-200,200,20)
for p in currents:
    y = odeint(rhs, y0, t, args=(p,))
    numTempPeaks = 0
    tempPeaks = [None]*2
    time = 2000

    # while numTempPeaks<2:
    #     if (y[time,0]>y[time-1,0] and y[time,0]>y[time+1,0]):
    #         tempPeaks[numTempPeaks]=time
    #         numTempPeaks += 1
    #     time += 1

    v = y[:, 0]
    v = v[t>50]
    if (np.max(v) < -30) and (np.min(v)>-90):
        spike_times = argrelmax(v, order=10)[0]
        f = 1000/np.mean(np.diff(t[spike_times]))
        #freqs.append(f)
    else:
        f = None
    freqs.append(f)
    #plt.figure(str(p))
    #plt.plot(t, y[:, 0])
    #plt.show()

# freqs = []
# for i in range(len(peakTimes)):
#     freqs.append(100000/(peakTimes[i][1]-peakTimes[i][0]))
#
# print("Frequencies: "+str(freqs))


plt.plot(currents,freqs)
