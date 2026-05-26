#%%lectores.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import os
from uncertainties import ufloat
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch, find_peaks
#%% Función para recortar a ciclos enteros
def recorte(t,v,frecuencia):
    '''
    Recorta un numero entero de periodos o ciclos,arrancando en fase 0 (campo max o campo min segun polaridad)
    Grafico: señal de muestra/calibracion, s/ fondo, s/ valor medio y recortadas a un numero entero de ciclos.
    '''
    #Numero de ciclos
    N_ciclos =  int(np.floor(t[-1]*frecuencia)) 
    
    #Indices ciclo
    indices_recorte = np.nonzero(np.logical_and(t>=0,t<N_ciclos/frecuencia))  
    
    #Si quisiera recortar ciclos a ambos lados
    # largo = indices_ciclo[-1][0]
    #if np.mod(largo,N_ciclos) == 0:
    # largo = largo - np.mod(largo,N_ciclos)
    #elif np.mod(largo,N_ciclos) <= 0.5:
        #largo = largo - np.mod(largo,N_ciclos)
    #else:
    # largo = largo + N_ciclos - np.mod(largo,N_ciclos)
    '''
    Recorto los vectores
    '''
    t_2 = t[indices_recorte]
    v_2 = v[indices_recorte]
    
    return t_2 , v_2 , N_ciclos
#%% sinusoide para ajuste de frecuencia

#%% Primer test: levanto y ploteo
dir_1 = 'data/t1/test_1.txt'
dir_2 = 'data/t1/test_2.txt'
dir_3 = 'data/t1/test_3.txt'
dir_4 = 'data/t1/test_4.txt'

frec_de_muestreo=100e6

data1=np.loadtxt(dir_1, skiprows=1,usecols=(0,1,2),dtype=float)
indx1=data1[:,0]
time1=data1[:,1] # ms
ch11=data1[:,2]  #mV

data2=np.loadtxt(dir_2, skiprows=1,usecols=(0,1,2),dtype=float)
indx2=data2[:,0]
time2=data2[:,1]
ch12=data2[:,2]

data3=np.loadtxt(dir_3, skiprows=1,usecols=(0,1,2),dtype=float)
indx3=data3[:,0]
time3=data3[:,1]
ch13=data3[:,2]

data4=np.loadtxt(dir_4, skiprows=1,usecols=(0,1,2),dtype=float)
indx4=data4[:,0]
time4=data4[:,1]
ch14=data4[:,2]
# %% 
fig,(ax,ax2,ax3)=plt.subplots(3,1,figsize=(10, 7),constrained_layout=True,sharex=True)
ax.plot(time1,ch11,label='test 1')
ax2.plot(time2,ch12,label='test 2')

ax3.plot(time3,ch13,label='test 3')
ax3.plot(time4,ch14,label='test 4')

for a in [ax,ax2,ax3]:
    a.set_ylabel('Voltage (V)')
    a.grid()
    a.legend()
ax3.set_xlabel('Time (s)')
#%% Datos raw del generador de señales

dir_gen1= 'data/gen/gen_1V_100kHz.txt'
data_gen1=np.loadtxt(dir_gen1, skiprows=1,usecols=(0,1,2),dtype=float)
indx_gen1=data_gen1[:,0]
time_gen1=data_gen1[:,1]  #ms
ch1_gen1=data_gen1[:,2]*1e-3   #mV

# time_gen1=time_gen1+time_gen1[0] #ms 

fig,ax=plt.subplots(figsize=(10, 5),constrained_layout=True)
ax.plot(indx_gen1,ch1_gen1,'.-',label='Generador 1V')
ax.set_ylabel('Voltage (mV)')
ax.grid()
ax.legend()
ax.set_xlabel('indx')
ax.set_title('señal 1 Vpp 100 kHz - indx')
#%% aplico escala temporal
SR = 100e6  # Samples per second
T = 1/SR  # Sample time interval
time = indx_gen1/SR  # Time vector

fig,ax=plt.subplots(figsize=(10, 5),constrained_layout=True)
ax.plot(time, ch1_gen1,'.-',label='Generador 1V')
ax.set_ylabel('Voltage (mV)')
ax.grid()   
ax.set_xlabel('Time (s)')
ax.set_title('señal 1 Vpp 100 kHz - tiempo')
#%% veo espectro de la señal del generador
dt = time[1] - time[0]
fs = 1/dt
N = len(ch1_gen1)

print(fs)
print(N)

ch1_gen1 = ch1_gen1 - np.mean(ch1_gen1) #resto offset

plt.figure(figsize=(8,4))

plt.plot(time, ch1_gen1)

plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')

plt.grid()
plt.show()
#%% FFT 
V = rfft(ch1_gen1)
# frecuencias
freq = rfftfreq(N, dt)

#Amplitudes físicas
A = 2*np.abs(V)/N
# corregir DC
A[0] /= 2

# corregir Nyquist
if N % 2 == 0:
    A[-1] /= 2
# pico principal
A0 = np.max(A)

# detección de picos en FFT
peaks, props = find_peaks(A,height=0.01*A0)
# mostrar
for p in peaks:
    print(f'{freq[p]:10.1f} Hz | {A[p]:.6f}')

plt.figure(figsize=(9,4))
plt.semilogy(freq, A)
#plt.plot(freq, A)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')

plt.xlim(0, 5e5)
plt.ylim(1e-4, 1e1)
plt.grid()
plt.show()
    
#%%  Welch para obtener PSD
from scipy.signal import welch

f_w, PSD = welch(
    ch1_gen1,
    fs=fs,
    window='hann',
    nperseg = 16384)

plt.figure(figsize=(9,4))
plt.semilogy(f_w, PSD)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [V²/Hz]')
plt.xlim(0, 5e5)
plt.grid()
plt.show()
#%% Comparo Welch y FFT
peaks, props = find_peaks(PSD,prominence=np.max(PSD)*1e-3)
for p in peaks:
    print(f'{f_w[p]:10.1f} Hz | PSD={PSD[p]:.3e}')
    
plt.figure(figsize=(9,4))
plt.semilogy(freq, A,label='FFT')
plt.semilogy(f_w, PSD,label='Welch')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [V²/Hz]')
plt.xlim(0, 5e5)
plt.grid()
plt.legend()
plt.show()

# %% Ahora analisis de medidas con RF nuevo
data_1 =np.loadtxt('data/t2/1.txt', skiprows=1,usecols=(0,1,2),dtype=float)
data_2 =np.loadtxt('data/t2/2.txt', skiprows=1,usecols=(0,1,2),dtype=float)
data_3 =np.loadtxt('data/t2/3.txt', skiprows=1,usecols=(0,1,2),dtype=float)
data_4 =np.loadtxt('data/t2/4.txt', skiprows=1,usecols=(0,1,2),dtype=float)
data_5 =np.loadtxt('data/t2/5.txt', skiprows=1,usecols=(0,1,2),dtype=float)
data_6 =np.loadtxt('data/t2/6.txt', skiprows=1,usecols=(0,1,2),dtype=float)

indx1,t1,v1=data_1[:,0],data_1[:,1],data_1[:,2]
indx2,t2,v2=data_2[:,0],data_2[:,1],data_2[:,2]
indx3,t3,v3=data_3[:,0],data_3[:,1],data_3[:,2]
indx4,t4,v4=data_4[:,0],data_4[:,1],data_4[:,2]
indx5,t5,v5=data_5[:,0],data_5[:,1],data_5[:,2]
indx6,t6,v6=data_6[:,0],data_6[:,1],data_6[:,2] 

v1=(v1-np.mean(v1))/1000
v2=(v2-np.mean(v2))/1000
v3=(v3-np.mean(v3))/1000
v4=(v4-np.mean(v4))/1000
v5=(v5-np.mean(v5))/1000
v6=(v6-np.mean(v6))/1000

SR = 100e6  # Samples per second
T = 1/SR  # Sample time interval

time1= indx1/SR  # Time vector
time2= indx2/SR  # Time vector
time3= indx3/SR  # Time vector
time4= indx4/SR  # Time vector
time5= indx5/SR  # Time vector
time6= indx6/SR  # Time vector

#%%
%matplotlib 
fig1,(a,b,c,d,e,f)=plt.subplots(6,1,figsize=(10, 10),constrained_layout=True,sharex=True,sharey=True)
a.plot(time1,v1,'-',label='Medida 1')
b.plot(time2,v2,'-',label='Medida 2')
c.plot(time3,v3,'-',label='Medida 3')
d.plot(time4,v4,'-',label='Medida 4')
e.plot(time5,v5,'-',label='Medida 5')
f.plot(time6,v6,'-',label='Medida 6')
for i in [a,b,c,d,e,f]:
    i.set_ylabel('Voltage (V)')
    i.grid()
    i.legend()
    i.set_xlim(0, )
f.set_xlabel('Time (s)')
#%% recorto a ciclos enteros 
# obtnego cruces por 0 para obtener frecucuencia
frecs=[]
from scipy.optimize import curve_fit
for t,v,f0 in zip([time1,time2,time3,time4,time5,time6],[v1,v2,v3,v4,v5,v6],[1e6,9e5,8e5,7e5,6e5,5.85e5]):
    popt,_ = curve_fit(sinusoide,t,v,p0=[10,f0,0])
    frecs.append(popt[1])
    print(f'{f0:9.1f} Hz | {popt[1]:9.1f} Hz')

#%% Recorto a ciclos enteros
t1,v1r,N_1=recorte(time1,v1,frecs[0])
t2,v2r,N_2=recorte(time2,v2,frecs[1])
t3,v3r,N_3=recorte(time3,v3,frecs[2])
t4,v4r,N_4=recorte(time4,v4,frecs[3])
t5,v5r,N_5=recorte(time5,v5,frecs[4])
t6,v6r,N_6=recorte(time6,v6,frecs[5])
# %%
fig2,(a,b,c,d,e,f)=plt.subplots(6,1,figsize=(10, 10),constrained_layout=True,sharex=True,sharey=True)
a.plot(t1,v1r,'-',label=f'f = {frecs[0]:.1f} Hz\nN = {N_1}')
b.plot(t2,v2r,'-',label=f'f = {frecs[1]:.1f} Hz\nN = {N_2}')
c.plot(t3,v3r,'-',label=f'f = {frecs[2]:.1f} Hz\nN = {N_3}')
d.plot(t4,v4r,'-',label=f'f = {frecs[3]:.1f} Hz\nN = {N_4}')
e.plot(t5,v5r,'-',label=f'f = {frecs[4]:.1f} Hz\nN = {N_5}')
f.plot(t6,v6r,'-',label=f'f = {frecs[5]:.1f} Hz\nN = {N_6}')

a.plot(time1,v1,'.',alpha=0.3,zorder=-3)
b.plot(time2,v2,'.',alpha=0.3,zorder=-3)
c.plot(time3,v3,'.',alpha=0.3,zorder=-3)
d.plot(time4,v4,'.',alpha=0.3,zorder=-3)
e.plot(time5,v5,'.',alpha=0.3,zorder=-3)
f.plot(time6,v6,'.',alpha=0.3,zorder=-3)


for i in [a,b,c,d,e,f]:
    i.set_ylabel('Voltage (mV)')
    i.grid()
    i.legend()
    i.set_xlim(0, )
f.set_xlabel('Time (s)')
# %% Ahora quiero realizar fft sobre las señales recortadas a ciclos enteros para obtener la frecuencia fundamental y sus armónicos, y comparar con la señal de calibración del generador.
# %% FFT + Welch sobre señal recortada a ciclos enteros

# =========================
# Señal
# =========================

t = t1
v = v1r

# remover offset DC
v = v - np.mean(v)

# =========================
# Parámetros temporales
# =========================

dt = t[1] - t[0]
fs = 1/dt
N = len(v)

print(f'fs = {fs:.3e} Hz')
print(f'N  = {N}')

# =========================
# FFT coherente
# =========================

V = rfft(v)

freq = rfftfreq(N, dt)

# amplitud física [V]
A = 2*np.abs(V)/N

# correcciones DC / Nyquist
A[0] /= 2

if N % 2 == 0:
    A[-1] /= 2

# =========================
# Detección picos FFT
# =========================

A0 = np.max(A)

peaks_fft, props_fft = find_peaks(A,height=0.01*A0)

print('\n=== Armónicos FFT ===')
for p in peaks_fft:
    print(
        f'{freq[p]/1e3:10.1f} kHz | '
        f'A = {A[p]:.6e} V')

# =========================
# Welch PSD
# =========================
nperseg = 4900
f_w, PSD = welch(v,fs=fs,window='hann',nperseg=nperseg)

# =========================
# Potencia total
# =========================

Ptot = np.trapz(PSD, f_w)

print('\n=========================')
print(f'Potencia total = {Ptot:.6e} V²')
print('=========================')

# =========================
# Picos Welch
# =========================

PSD0 = np.max(PSD)

peaks_w, props_w = find_peaks(PSD,prominence=PSD0*1e-3)

print('\n=== Picos Welch PSD ===')
for p in peaks_w:
    print(
        f'{f_w[p]/1e3:10.1f} kHz | '
        f'PSD = {PSD[p]:.3e} V²/Hz'
    )
# =========================
# Potencia por banda Welch
# =========================

BW_bins = 10     # integrar ±10 bins

print('\n=== Potencia por banda ===')

for p in peaks_w:

    # límites índices
    i0 = max(0, p-BW_bins)
    i1 = min(len(PSD)-1, p+BW_bins)

    # integración
    P_band = np.trapz(
        PSD[i0:i1+1],
        f_w[i0:i1+1]
    )

    perc = 100 * P_band / Ptot

    print(
        f'{f_w[p]/1e3:10.1f} kHz | '
        f'P = {P_band:.3e} V² | '
        f'{perc:.2f} %'
    )

# =========================
# Figura resumen
# =========================

fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(10,9),
    constrained_layout=True)

# -------- señal temporal --------
ax1.set_title('Señal temporal',loc='left')
ax1.plot(t,v,label=f'f = {frecs[0]:.1f} Hz\nN = {N_1}')
ax1.set_ylabel('Voltage [V]')
ax1.set_xlabel('Time [s]')

# -------- FFT --------
ax2.set_title('FFT coherente',loc='left')
ax2.semilogy(freq/1e3,A,label='FFT')
ax2.semilogy(freq[peaks_fft]/1e3,A[peaks_fft],'ro')
ax2.set_ylabel('Amplitude [V]')
ax2.set_xlabel('Frequency [kHz]')
ax2.set_xlim(0,2e4)
ax2.set_ylim(1e-6,2e1)

# -------- Welch --------
ax3.set_title('Welch PSD',loc='left')
ax3.semilogy(f_w/1e3,PSD,label='Welch PSD')
ax3.semilogy(f_w[peaks_w]/1e3,PSD[peaks_w],'ro')
ax3.set_ylabel('PSD [V²/Hz]')
ax3.set_xlabel('Frequency [kHz]')
ax3.set_xlim(0,2e4)
ax3.set_ylim(1e-12,2e1)

for a in [ax1, ax2, ax3]:
    a.grid()
    a.legend()
plt.show()
# %% Sistematizo para repetir analisis
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch, find_peaks
import numpy as np
import matplotlib.pyplot as plt


def analizar_senal(
    t,
    v,
    nombre='', Nciclos=None,
    fmax_plot=2e7,
    nperseg=4900,
    BW_bins=10,
    fft_threshold=0.01,
    welch_prominence=1e-3,
    plot=True
):

    # =========================
    # remover DC
    # =========================

    v = v - np.mean(v)

    # =========================
    # parámetros
    # =========================

    dt = t[1] - t[0]
    fs = 1/dt
    N = len(v)

    print('\n=================================')
    print(nombre)
    print('=================================')

    print(f'fs = {fs:.3e} Hz')
    print(f'N  = {N}')
    print(f'Num ciclos = {Nciclos}')

    # =========================
    # FFT
    # =========================

    V = rfft(v)

    freq = rfftfreq(N, dt)

    A = 2*np.abs(V)/N

    A[0] /= 2

    if N % 2 == 0:
        A[-1] /= 2

    # =========================
    # picos FFT
    # =========================

    A0 = np.max(A)

    peaks_fft, props_fft = find_peaks(
        A,
        height=fft_threshold*A0
    )

    print('\n=== Armónicos FFT ===')

    for p in peaks_fft:

        print(
            f'{freq[p]/1e3:10.1f} kHz | '
            f'A = {A[p]:.6e} V'
        )

    # =========================
    # Welch
    # =========================

    f_w, PSD = welch(
        v,
        fs=fs,
        window='hann',
        nperseg=nperseg
    )

    # =========================
    # potencia total
    # =========================

    Ptot = np.trapz(PSD, f_w)

    print('\n=========================')
    print(f'Potencia total = {Ptot:.6e} V²')
    print('=========================')

    # =========================
    # picos Welch
    # =========================

    PSD0 = np.max(PSD)

    peaks_w, props_w = find_peaks(
        PSD,
        prominence=PSD0*welch_prominence
    )

    print('\n=== Picos Welch PSD ===')

    for p in peaks_w:

        print(
            f'{f_w[p]/1e3:10.1f} kHz | '
            f'PSD = {PSD[p]:.3e} V²/Hz'
        )

    # =========================
    # potencia por banda
    # =========================

    print('\n=== Potencia por banda ===')

    bandas = []

    for p in peaks_w:

        i0 = max(0, p-BW_bins)
        i1 = min(len(PSD)-1, p+BW_bins)

        P_band = np.trapz(
            PSD[i0:i1+1],
            f_w[i0:i1+1]
        )

        perc = 100 * P_band / Ptot

        bandas.append({
            'freq': f_w[p],
            'power': P_band,
            'percent': perc
        })

        print(
            f'{f_w[p]/1e3:10.1f} kHz | '
            f'P = {P_band:.3e} V² | '
            f'{perc:.2f} %'
        )

    # =========================
    # plots
    # =========================

    if plot:

        fig, (ax1, ax2, ax3) = plt.subplots(3,1,
            figsize=(10,9),constrained_layout=True)

        # temporal
        ax1.set_title(f'{nombre} - Señal temporal',loc='left')
        ax1.plot(t,v,label=f'f = {freq[peaks_fft[0]]/1000:.1f} kHz\nN ciclos = {Nciclos}')

        ax1.set_ylabel('Voltage [V]')
        ax1.set_xlabel('Time [s]')

        # FFT
        ax2.set_title('FFT coherente',loc='left')
        ax2.semilogy(freq/1e3,A,label='FFT')

        ax2.semilogy(freq[peaks_fft]/1e3,A[peaks_fft],'ro')

        ax2.set_ylabel('Amplitude [V]')
        ax2.set_xlabel('Frequency [kHz]')

        ax2.set_xlim(0,1.1*fmax_plot/1e3)
        ax2.set_ylim(1e-6,2.5e1)

        # Welch

        ax3.set_title(
            'Welch PSD',
            loc='left'
        )

        ax3.semilogy(
            f_w/1e3,
            PSD,
            label='Welch PSD'
        )

        ax3.semilogy(
            f_w[peaks_w]/1e3,
            PSD[peaks_w],
            'ro'
        )

        ax3.set_ylabel('PSD [V²/Hz]')
        ax3.set_xlabel('Frequency [kHz]')

        ax3.set_xlim(0,fmax_plot/1e3)
        ax3.set_ylim(1e-12,2e1)

        for a in [ax1, ax2, ax3]:

            a.grid()
            a.legend()

        plt.show()

    # =========================
    # devolver resultados
    # =========================

    return {

        'freq': freq,
        'A': A,

        'f_w': f_w,
        'PSD': PSD,

        'Ptot': Ptot,

        'peaks_fft': peaks_fft,
        'peaks_w': peaks_w,

        'bandas': bandas
    }

# %% implemento 
res1 = analizar_senal(t1, v1r, Nciclos=N_1, nombre='Señal 1')
# %%
res2 = analizar_senal(t2, v2r, Nciclos=N_2, nombre='Señal 2')
# %%
res3 = analizar_senal(t3, v3r, Nciclos=N_3, nombre='Señal 3')
# %%
res4 = analizar_senal(t4, v4r, Nciclos=N_4, nombre='Señal 4')
# %%
res5 = analizar_senal(t5, v5r, Nciclos=N_5, nombre='Señal 5')
# %%
res6 = analizar_senal(t6, v6r, Nciclos=N_6, nombre='Señal 6')
# %%
