import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
from scipy.signal import find_peaks

# kezdeti értékek beállítása
y0 = [0.5, 0.2, 0, 0.1, 0.01, 1.4, 0.037, 0.046, 0.13, 3, 0.1, 0.1, 0.001, 0.1, 0.1, 2]
# [N, Na, Na_n, M_F, F, M=1.4, CP=0.037, CP2=0.046, TF=0.13, K, B, B_p, B_N, M_Ax, A, D]

t = np.linspace(0, 12000, num=2000)

# paraméterek megadása
v_sN = 0.23  # Maximum rate of Notch synthesis
v_dN = 2.82  # Maximum rate of Notch degradation
K_dN = 1.4  # Michaelis constant for Notch degradation
k_c = 3.45  # Apparent ﬁrst-order rate constant for Notch (N) cleavage into NICD (Na)
V_dNa = 0.01  # Maximum rate of NICD degradation
K_dNa = 0.001  # Michaelis constant for NICD degradation
V_dNan = 0.1  # Maximum rate of nuclear NICD degradation
K_dNan = 0.001  # Michaelis constant for nuclear NICD degradation
K_IF = 0.5  # Threshold constant for inhibition by Lunatic Fringe of Notch cleavage into NICD
k_t1 = 0.1  # Apparent ﬁrst-order rate constant for NICD entry into the nucleus
k_t2 = 0.1  # Apparent ﬁrst-order rate constant for NICD exit from the nucleus
v_sF = 3  # Maximum rate of Lunatic fringe gene transcription
K_A = 0.05  # Threshold constant for activation of Lunatic fringe gene transcription by nuclear NICD
v_mF = 1.92  # Maximum rate of Lunatic fringe mRNA degradation
K_dmF = 0.768  # Michaelis constant for Lunatic fringe mRNA degradation
k_sF = 0.3  # Apparent ﬁrst-order rate constant for Lunatic Fringe protein synthesis
v_dF = 0.39  # Maximum rate of Lunatic Fringe protein degradation
K_dF = 0.37  # Michaelis constant for Lunatic Fringe protein degradation
K_IG1 = 2.5  # Inhibition constant for Gsk inhibition of Lunatic fringe transcription induced by NICD
j = 2  # Hill-coefficient
p = 2  # Hill-coefficient
epsilon = 0.19  # ez volt a Goldbeter cikkben. a periódus 11.6

# parameters for the messenger RNA of the clock protein (Per mRNA)
nn = 2
J = 0.3
k_ms = 1/60
k_md = 0.1/60

# parameters for the clock protein (PER)
k_cps = 0.5/60
k_cpd = 0.525/60
k_a = 100/60
k_d = 0.01/60
k_p1 = 10/60
J_p = 0.05

# parameters for the dimer clock protein (PER/PER)
k_cp2d = 0.0525/60
k_p2 = 0.1/60

# parameters for the transcription factor (BMAL1/CLK) and its inhibition
k_icd = 0.01/60
k_ica = 20/60
TF_tot = 0.5 # total BMAL1/CLK concentration

a_1 = 1.8  # Bimolecular rate constant for binding of Gsk3 to Axin2
d_1 = 0.1  # Rate constant for dissociation of Gsk3–Axin2 complex
v_sB = 0.087  # Maximum rate of b-catenin synthesis
k_t3 = 0.7  # Apparent ﬁrst-order rate constant for b-catenin entry into the nucleus
k_t4 = 1.5  # Apparent ﬁrst-order rate constant for b-catenin exit from the nucleus
V_MK = 5.08  # Maximum rate of phosphorylation of b-catenin by the kinase Gsk3
# D = 2           # Dishevelled (Dsh) protein concentration
K_t = 3  # Total Gsk3 concentration
K_ID = 0.5  # Constant of inhibition by Dsh of b-catenin phosphorylation by the Axin2–Gsk3 destruction complex
K_1 = 0.28  # Michaelis constant for b-catenin phosphorylation by the Axin2–Gsk complex
V_MP = 1  # Maximum rate of dephosphorylation of b-catenin
K_2 = 0.03  # Michaelis constant for b-catenin dephosphorylation
k_d1 = 0  # Apparent ﬁrst-order rate constant for degradation of unphosphorylated b-catenin
k_d2 = 7.062  # Apparent ﬁrst-order rate constant for degradation of phosphorylated b-catenin
v_0 = 0.06  # Basal rate of transcription of the Axin2 gene
v_MB = 1.64  # Maximum rate of transcription of the Axin2 gene induced by nuclear b-catenin
K_aB = 0.7  # Threshold constant for induction by nuclear b-catenin of Axin2 gene transcription
v_md = 0.8  # Maximum rate of degradation of Axin2 mRNA
K_md = 0.48  # Michaelis constant for degradation of Axin2 mRNA
v_MXa = 0.5  # Maximum rate of transcription of the Axin2 gene induced by factor X_a
K_aXa = 0.05  # Threshold constant for induction by factor Xa of Axin2 gene transcription
k_sAx = 0.02  # Apparent ﬁrst-order rate constant for synthesis of Axin2 protein
v_dAx = 0.6  # Threshold constant for induction by factor Xa of Axin2 gene transcription
K_dAx = 0.63  # Michaelis constant for degradation of Axin2 protein
n = 2  # Hill coefﬁcient
m = 2  # Hill coefﬁcient
theta = 1.5  # Scaling factor for Wnt oscillator

k_deg = 0.6   # Dsh degradation rate
coupling2 = 0
coupling1 = 0

params = [v_sN, v_dN, K_dN, k_c, V_dNa, K_dNa, V_dNan, K_dNan, K_IF, k_t1, k_t2, v_sF, K_A, v_mF, K_dmF, k_sF, v_dF,
          K_dF, K_IG1, j, p, epsilon, nn, J, k_ms, k_md, k_cps, k_cpd, k_a, k_d, k_p1, J_p, k_cp2d, k_p2,
          k_icd, k_ica, TF_tot, a_1, d_1, v_sB, k_t3, k_t4, V_MK, K_t, K_ID, K_1, V_MP, K_2, k_d1, k_d2, v_0, v_MB,
          K_aB, v_md, K_md, k_sAx, v_dAx, K_dAx, n, theta, k_deg, coupling2, coupling1]

# ode-solver
def sim(variables, t, params):
    N = variables[0]  # Notch protein
    Na = variables[1]  # cytosolic NICD
    Na_n = variables[2]  # nuclear NICD
    M_F = variables[3]  # Lunatic Fringe mRNA
    F = variables[4]  # Lunatic Fringe protein
    M = variables[5] # Messenger RNA of the clock proteins (Per mRNA):
    CP = variables[6] # Monomer clock proteins (PER):
    CP2 = variables[7] # Dimer form of clock proteins (PER/PER):
    TF = variables[8] # Transcription factor (BMAL1/CLK):
    K = variables[9]  # concentrations of the free kinase Gsk3
    B = variables[10]  # concentrations of the cytosolic form of nonphosphorylated b-catenin
    B_p = variables[11]  # concentrations of the cytosolic form of phosphorylated b-catenin
    B_N = variables[12]  # concentrations of the nuclear b-catenin
    M_Ax = variables[13]  # concentrations of the Axin2 mRNA
    A = variables[14]  # concentrations of the Axin2 protein
    D = variables[15]  # Dsh concentration

    v_sN = params[0]
    v_dN = params[1]
    K_dN = params[2]
    k_c = params[3]
    V_dNa = params[4]
    K_dNa = params[5]
    V_dNan = params[6]
    K_dNan = params[7]
    K_IF = params[8]
    k_t1 = params[9]
    k_t2 = params[10]
    v_sF = params[11]
    K_A = params[12]
    v_mF = params[13]
    K_dmF = params[14]
    k_sF = params[15]
    v_dF = params[16]
    K_dF = params[17]
    K_IG1 = params[18]
    j = params[19]
    p = params[20]
    epsilon = params[21]

    nn = params[22]
    J = params[23]
    k_ms = params[24]
    k_md = params[25]
    k_cps = params[26]
    k_cpd = params[27]
    k_a = params[28]
    k_d = params[29]
    k_p1 = params[30]
    J_p = params[31]
    k_cp2d = params[32]
    k_p2 = params[33]
    k_icd = params[34]
    k_ica = params[35]
    TF_tot = params[36]

    a_1 = params[37]
    d_1 = params[38]
    v_sB = params[39]
    k_t3 = params[40]
    k_t4 = params[41]
    V_MK = params[42]
    K_t = params[43]
    K_ID = params[44]
    K_1 = params[45]
    V_MP = params[46]
    K_2 = params[47]
    k_d1 = params[48]
    k_d2 = params[49]
    v_0 = params[50]
    v_MB = params[51]
    K_aB = params[52]
    v_md = params[53]
    K_md = params[54]
    k_sAx = params[55]
    v_dAx = params[56]
    K_dAx = params[57]
    n = params[58]
    theta = params[59]
    k_deg = params[60]
    coupling2 = params[61]
    coupling1 = params[62]

    dNdt = epsilon * (v_sN - v_dN * N / (K_dN + N) - k_c * N * K_IF ** j / (K_IF ** j + F ** j))
    dNadt = epsilon * (k_c * N * K_IF ** j / (K_IF ** j + F ** j) - V_dNa * Na / (K_dNa + Na) - k_t1 * Na + k_t2 * Na_n)
    dNa_ndt = epsilon * (k_t1 * Na - k_t2 * Na_n - V_dNan * Na_n / (K_dNan + Na_n))
    dM_Fdt = epsilon * ((v_sF * K_IG1 / (K_IG1 + coupling2 * K)) * Na_n ** p / (K_A ** p + Na_n ** p) - v_mF * M_F / (K_dmF + M_F))
    dFdt = epsilon * (k_sF * M_F - v_dF * F / (K_dF + M_F))

    dMdt = k_ms * (TF ** nn) / (J ** nn + TF ** nn) - k_md * M
    dCPdt = k_cps * M - k_cpd * CP - 2 * k_a * CP ** 2 + 2 * k_d * CP2 - k_p1 * CP / (J_p + CP + 2 * CP2 + 2 * (TF_tot - TF))
    dCP2dt = k_a * CP ** 2 - k_d * CP2 - k_cp2d * CP2 + k_icd * (TF_tot - TF) - k_ica * CP2 * TF - k_p2 * CP2 / (J_p + CP + 2 * CP2 + 2 * (TF_tot - TF))
    dTFdt = k_cp2d * (TF_tot - TF) + k_icd * (TF_tot - TF) - k_ica * TF * CP2 + k_p2 * (TF_tot - TF) / (J_p + CP + 2 * CP2 + 2 * (TF_tot - TF))

    dKdt = theta * (d_1 * (K_t - K) - a_1 * A * K)
    dBdt = theta * (v_sB - V_MK * K_ID * B * (K_t - K) / ((K_ID + D) * (K_1 + B) * K_t) + V_MP * B_p / (K_2 + B_p) - k_t3 * B + k_t4 * B_N)
    dB_pdt = theta * (V_MK * K_ID * B * (K_t - K) / ((K_ID + D) * (K_1 + B) * K_t) - V_MP * B_p / (K_2 + B_p) - k_d2 * B_p)
    dB_Ndt = theta * (k_t3 * B - k_t4 * B_N)
    dM_Axdt = theta * (v_0 + v_MB * B_N ** n / (K_aB ** n + B_N ** n) - v_md * M_Ax / (K_md + M_Ax))
    dAdt = theta * (k_sAx * M_Ax - v_dAx * A / (K_dAx + A) - a_1 * A * K + d_1 * (K_t - K))
    dDdt = 1.2 / (1 + coupling1) + coupling1 * 5 * TF - k_deg * D

    return ([dNdt, dNadt, dNa_ndt, dM_Fdt, dFdt, dMdt,dCPdt,dCP2dt,dTFdt, dKdt, dBdt, dB_pdt, dB_Ndt, dM_Axdt, dAdt, dDdt])


# az egyes konkrét szimulációk kódjai, csak a megfelelőt kell kikommentelni

"""
# 1a
coupling2 = 0
coupling1 = 0
y = odeint(sim, y0, t, args=(params,))
f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)

peaks, _ = find_peaks(y[:, 3])
peak_diff = np.diff(peaks)
period = np.mean(peak_diff)

periods = np.zeros_like(y[:, 3])
for i, p in enumerate(peaks[:-1]):
    periods[p:peaks[i + 1]] = peak_diff[i]

# Set time differences to zero at peak indices
# np.put(peak_diff, peaks, 0)

xt = np.linspace(0, 12000, num=peak_diff.size)

line1, = ax1.plot(t, y[:, 3], color="b", label="M_F")
ax1.set_ylabel("concentration")
ax1.set_xlabel("time [m]")
line2, = ax2.plot(t, periods, label="peak_diff")
plt.ylabel("period [m]")
plt.xlabel("time [m]")

ax2.legend(handles=[line1, line2])

plt.show()
# veg
"""

"""
# 1b
coupling2 = 1
coupling1 = 1
y = odeint(sim, y0, t, args=(params,))
f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)

peaks, _ = find_peaks(y[:, 3])
peak_diff = np.diff(peaks)
period = np.mean(peak_diff)

periods = np.zeros_like(y[:, 3])
for i, p in enumerate(peaks[:-1]):
    periods[p:peaks[i + 1]] = peak_diff[i]

# Set time differences to zero at peak indices
# np.put(peak_diff, peaks, 0)

xt = np.linspace(0, 12000, num=peak_diff.size)

line1, = ax1.plot(t, y[:, 3], color="b", label="M_F")
ax1.set_ylabel("concentration")
ax1.set_xlabel("time [m]")
line2, = ax2.plot(t, periods, label="peak_diff")
plt.ylabel("period [m]")
plt.xlabel("time [m]")

ax2.legend(handles=[line1, line2])

plt.show()
"""

"""
# 2a
coupling2 = 0
coupling1 = 0
y = odeint(sim, y0, t, args=(params,))
f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)

peaks, _ = find_peaks(y[:, 10])
peak_diff = np.diff(peaks)
period = np.mean(peak_diff)

periods = np.zeros_like(y[:, 10])
for i, p in enumerate(peaks[:-1]):
    periods[p:peaks[i + 1]] = peak_diff[i]

# Set time differences to zero at peak indices
# np.put(peak_diff, peaks, 0)

xt = np.linspace(0, 12000, num=peak_diff.size)

line1, = ax1.plot(t, y[:, 10], color="b", label="M_Ax")
ax1.set_ylabel("concentration")
ax1.set_xlabel("time [m]")
line2, = ax2.plot(t, periods, label="peak_diff")
plt.ylabel("period [m]")
plt.xlabel("time [m]")

ax2.legend(handles=[line1, line2])

plt.show()
"""

"""
# 2b
coupling2 = 1
coupling1 = 1
y = odeint(sim, y0, t, args=(params,))
f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)

peaks, _ = find_peaks(y[:, 3])
peak_diff = np.diff(peaks)
period = np.mean(peak_diff)

periods = np.zeros_like(y[:, 3])
for i, p in enumerate(peaks[:-1]):
    periods[p:peaks[i + 1]] = peak_diff[i]

# Set time differences to zero at peak indices
# np.put(peak_diff, peaks, 0)

xt = np.linspace(0, 12000, num=peak_diff.size)

line1, = ax1.plot(t, y[:, 3], color="b", label="M_Ax")
ax1.set_ylabel("concentration")
ax1.set_xlabel("time [m]")
line2, = ax2.plot(t, periods, label="peak_diff")
plt.ylabel("period [m]")
plt.xlabel("time [m]")

ax2.legend(handles=[line1, line2])

plt.show()
"""

"""
# 3
ytf = np.empty([0,2000])
y2d = np.empty([0,2000])
ymax = np.empty([0,2000])

for i in range(0, 10, 3):
    params[62] = i/10
    y = odeint(sim, y0, t, args=(params,))
    ymax = np.vstack([ymax, y[:, 10]])
    ytf = np.vstack([ytf, y[:, 9]])
    y2d = np.vstack([y2d, y[:, 15]])
    print(y2d.shape)

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=False)

line1, = ax1.plot(t , ytf[0,:], color="b", label="dTFdt")
line2, = ax1.plot(t , y2d[0,:], color="g", label="dDdt")
line3, = ax1.plot(t , ymax[0,:], color="r", label="dM_Axdt")
line4, = ax2.plot(t , ytf[1,:], color="b")
line5, = ax2.plot(t , y2d[1,:], color="g")
line6, = ax2.plot(t , ymax[1,:], color="r")
line7, = ax3.plot(t , ytf[2,:], color="b")
line8, = ax3.plot(t , y2d[2,:], color="g")
line9, = ax3.plot(t , ymax[2,:], color="r")
line10, = ax4.plot(t , ytf[3,:], color="b")
line11, = ax4.plot(t , y2d[3,:], color="g")
line12, = ax4.plot(t , ymax[3,:], color="r")

#line1, = ax1.plot(t , fft(y2d[0,:]), color="b",label="D")
#line2, = ax2.plot(t , fft(y2d[1,:]), color="r",label="D")
#line3, = ax3.plot(t , fft(y2d[2,:]), color="g",label="D")
#line4, = ax4.plot(t , fft(y2d[3,:]), color="c",label="D")
#line5, = ax5.plot(t , fft(y2d[4,:]), color="m",label="D")

ax1.set_xlabel("time [m] (coupling1 = 0)")
ax2.set_xlabel("time [m] (coupling1 = 0.3)")
ax3.set_xlabel("time [m] (coupling1 = 0.6)")
ax4.set_xlabel("time [m] (coupling1 = 0.9)")

f.legend()

plt.show()
"""

"""
# 4.
# Ez a k_1 és a coupling2 növelésvel lentebb kiválasztható jel periódusát plotolja 2dben
krange = np.linspace(0, 1, num=10);
crange = np.linspace(0, 1, num=10);

arr = np.empty([10, 10])

for i in range(10):
    for j in range(10):
        params[61] = krange[i]
        params[62] = crange[j]
        y = odeint(sim,y0,t,args=(params,))
        peaks, _ = find_peaks(y[:,3])
        # Calculate the time difference between peaks
        peak_diff = np.diff(peaks)
        # Calculate the period as the average of the time differences
        period = np.mean(peak_diff)
        arr[i,j] = period
        print(i)
        print(j)


# create a 2D array

# create x and y coordinates
x = np.arange(0, arr.shape[0])
y = np.arange(0, arr.shape[1])
X, Y = np.meshgrid(x, y)

# plot the array in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlabel("M_F concentration")
ax.plot_surface(X, Y, arr)
plt.xlabel("coupling2")
plt.ylabel("coupling1")
plt.show()
# vége
"""
