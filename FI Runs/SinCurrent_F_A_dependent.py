import numpy as np
from numpy import exp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from scipy.signal import argrelmax


def rhs(y, t, f, am):
    v, b, g, a, h, m, n, q = y
    I = am * np.sin(f * 2 * np.pi * t)
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

def getFreqs(y0,t,fs,ams):  # initial conditions, times, frequency, amplitude
    freqs = []

    for f in fs:
        f_row=[]
        for am in ams:

            y = odeint(rhs, y0, t, args=(f, am), atol=1e-6, rtol=1e-6)

            v = y[:, 0]
            v = v[t > 50]

            if (np.max(v) < -30) and (np.min(v) > -90):
                spike_times = argrelmax(v, order=10)[0]
                spike_times = spike_times[v[spike_times] > -55]
                freq = 1000 / np.mean(np.diff(t[spike_times]))
            else:
                freq = None

            f_row.append(freq)
        freqs.append(f_row)

    return freqs


y0 = [0, 0, 0, 0, 0, 0, 0, 0]
t = np.arange(0, 100, 0.01)  # t0, t1, step

fs = range(300,361,15)  # frequencies of sin function
ams = range(0,200,20)  # amplitude of sin function

freqs = getFreqs(y0,t,fs,ams)

plt.figure()

fig = plt.subplot()

im = fig.imshow(freqs)
fig.set_xticks(np.arange(len(ams)))
fig.set_yticks(np.arange(len(fs)))
fig.set_xticklabels(ams)
fig.set_yticklabels(fs)

for edge, spine in fig.spines.items():
    spine.set_visible(False)

fig.set_xticks(np.arange(len(ams)+1)-.5, minor=True)
fig.set_yticks(np.arange(len(fs)+1)-.5, minor=True)
fig.grid(which="minor", color="w", linestyle='-', linewidth=3)
fig.tick_params(which="minor", bottom=False, left=False)

cbar = fig.figure.colorbar(im, ax=fig)
cbar.ax.set_ylabel('Frequency', rotation=-90, va="bottom")

plt.show()
