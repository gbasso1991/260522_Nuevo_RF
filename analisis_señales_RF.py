#%%lectores.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import os
from uncertainties import ufloat
#%%Funciones
#%% PLOTEADOR CICLOS PROMEDIO
def plot_ciclos_promedio(directorio):
    # Buscar recursivamente todos los archivos que coincidan con el patrón
    """
    Buscar recursivamente todos los archivos que coincidan con el patrón
    *ciclo_promedio*.txt en el directorio especificado y sus subdirectorios,
    y graficar sus ciclos de histéresis.

    Parameters
    ----------
    directorio : str
        Directorio donde se busca recursivamente

    Returns
    -------
    None
    """
    archivos = glob(os.path.join(directorio, '**', '*ciclo_promedio*.txt'), recursive=True)
    archivos.sort()
    if not archivos:
        print(f"No se encontraron archivos '*ciclo_promedio.txt' en {directorio} o sus subdirectorios")
        return
    fig,ax=plt.subplots(figsize=(8, 6),constrained_layout=True)
    for archivo in archivos:
        try:
            # Leer los metadatos (primeras líneas que comienzan con #)
            metadatos = {}
            with open(archivo, 'r') as f:
                for linea in f:
                    if not linea.startswith('#'):
                        break
                    if '=' in linea:
                        clave, valor = linea.split('=', 1)
                        clave = clave.replace('#', '').strip()
                        metadatos[clave] = valor.strip()

            # Leer los datos numéricos
            datos = np.loadtxt(archivo, skiprows=9)  # Saltar las 8 líneas de encabezado/metadatos

            tiempo = datos[:, 0]
            campo = datos[:, 3]  # Campo en kA/m
            magnetizacion = datos[:, 4]  # Magnetización en A/m

            # Crear etiqueta para la leyenda
            nombre_base = os.path.split(archivo)[-1].split('_')[1]
            #os.path.basename(os.path.dirname(archivo))  # Nombre del subdirectorio
            etiqueta = f"{nombre_base}"

            # Graficar

            ax.plot(campo, magnetizacion, label=etiqueta)

        except Exception as e:
            print(f"Error procesando archivo {archivo}: {str(e)}")
            continue

    plt.xlabel('H (kA/m)')
    plt.ylabel('M (A/m)')
    plt.title(f'Comparación de ciclos de histéresis {os.path.split(directorio)[-1]}')
    plt.grid(True)
    plt.legend()  # Leyenda fuera del gráfico
    plt.savefig('comparativa_ciclos_'+os.path.split(directorio)[-1]+'.png',dpi=300)
    plt.show()
#%% LECTOR RESULTADOS
def lector_resultados(path):
    """
    Lee archivo resultados.txt devuelve los datos y metadatos

    Parameters
    ----------
    path : str
        Ruta del archivo a leer

    Returns
    -------
    meta : dict
        Diccionario con metadatos del archivo
    files : numpy.ndarray
        Nombres de los archivos procesados
    time : numpy.ndarray
        Tiempos de medición en minutos
    temperatura : numpy.ndarray
        Temperaturas de medición en ºC
    Mr : numpy.ndarray
        Remanencia en A/m
    Hc : numpy.ndarray
        Coercitividad en kA/m
    campo_max : numpy.ndarray
        Campo máximo en kA/m
    mag_max : numpy.ndarray
        Magnetización máxima en A/m
    xi_M_0 : numpy.ndarray
        Magnetización en A/m a 0 K
    frecuencia_fund : numpy.ndarray
        Frecuencia fundamental en Hz
    magnitud_fund : numpy.ndarray
        Magnitud de la frecuencia fundamental en A/m
    dphi_fem : numpy.ndarray
        Ángulo de fase de la frecuencia fundamental en rad
    SAR : numpy.ndarray
        SAR en W/g
    tau : numpy.ndarray
        Constante de tiempo de relajación en s
    N : numpy.ndarray
        Número de datos utilizados para la ajuste"""
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 20 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                # Patrón para valores con incertidumbre (ej: 331.45+/-6.20 o (9.74+/-0.23)e+01)
                match_uncertain = re.search(r'(.+)_=_\(?([-+]?\d+\.\d+)\+/-([-+]?\d+\.\d+)\)?(?:e([+-]\d+))?', line)
                if match_uncertain:
                    key = match_uncertain.group(1)[2:]  # Eliminar '# ' al inicio
                    value = float(match_uncertain.group(2))
                    uncertainty = float(match_uncertain.group(3))
                    
                    # Manejar notación científica si está presente
                    if match_uncertain.group(4):
                        exponent = float(match_uncertain.group(4))
                        factor = 10**exponent
                        value *= factor
                        uncertainty *= factor
                    
                    meta[key] = ufloat(value, uncertainty)
                else:
                    # Patrón para valores simples (sin incertidumbre)
                    match_simple = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                    if match_simple:
                        key = match_simple.group(1)[2:]
                        value = float(match_simple.group(2))
                        meta[key] = value
                    else:
                        # Capturar los casos con nombres de archivo
                        match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                        if match_files:
                            key = match_files.group(1)[2:]
                            value = match_files.group(2)
                            meta[key] = value

    # Leer los datos del archivo (esta parte permanece igual)
    data = pd.read_table(path, header=15,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    """
    Lee un archivo de texto y devuelve los datos de los ciclos y metadatos

    Parameters
    ----------
    filepath : str
        Ruta del archivo a leer

    Returns
    -------
    t : numpy.ndarray
        Tiempos de los ciclos
    H_Vs : numpy.ndarray
        Campo de los ciclos en Vs
    M_Vs : numpy.ndarray
        Magnetizacion de los ciclos en Vs
    H_kAm : numpy.ndarray
        Campo de los ciclos en kA/m
    M_Am : numpy.ndarray
        Magnetizacion de los ciclos en A/m
    metadata : dict
        Diccionario con metadatos del archivo
    """
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype={'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata


#%% Primero levanto y ploteo
dir_1 = 'data/test_1.txt'
dir_2 = 'data/test_2.txt'
dir_3 = 'data/test_3.txt'
dir_4 = 'data/test_4.txt'

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
%matplotlib inline
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
#%% Datos del generador
%matplotlib
dir_gen1= 'data/gen_1V_100kHz.txt'
data_gen1=np.loadtxt(dir_gen1, skiprows=1,usecols=(0,1,2),dtype=float)
indx_gen1=data_gen1[:,0]
time_gen1=data_gen1[:,1]  #ms
ch1_gen1=data_gen1[:,2]   #mV

# time_gen1=time_gen1+time_gen1[0] #ms 

fig,ax=plt.subplots(figsize=(10, 5),constrained_layout=True)
ax.plot(indx_gen1,ch1_gen1,'.-',label='Generador 1V')
ax.set_ylabel('Voltage (mV)')
ax.grid()
ax.legend()
                                                 
ax.set_xlabel('indx')
# %%

SR = 100e6  # Samples per second
T = 1/SR  # Sample time interval
time = indx_gen1/SR  # Time vector

fig,ax=plt.subplots(figsize=(10, 5),constrained_layout=True)
ax.plot(time, ch1_gen1,'.-',label='Generador 1V')
ax.set_ylabel('Voltage (mV)')
ax.grid()   
ax.set_xlabel('Time (s)')
# %% Nueva serie de medidas con RF nuevo
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

v1=v1/1000
v2=v2/1000
v3=v3/1000
v4=v4/1000
v5=v5/1000
v6=v6/1000

SR = 100e6  # Samples per second
T = 1/SR  # Sample time interval

time1= indx1/SR  # Time vector
time2= indx2/SR  # Time vector
time3= indx3/SR  # Time vector
time4= indx4/SR  # Time vector
time5= indx5/SR  # Time vector
time6= indx6/SR  # Time vector

#%%
fig1,(a,b,c,d,e,f)=plt.subplots(6,1,figsize=(10, 10),constrained_layout=True,sharex=True,sharey=True)
a.plot(time1,v1,'-',label='Medida 1')
b.plot(time2,v2,'-',label='Medida 2')
c.plot(time3,v3,'-',label='Medida 3')
d.plot(time4,v4,'-',label='Medida 4')
e.plot(time5,v5,'-',label='Medida 5')
f.plot(time6,v6,'-',label='Medida 6')
for i in [a,b,c,d,e,f]:
    i.set_ylabel('Voltage (mV)')
    i.grid()
    i.legend()
    i.set_xlim(0, )
f.set_xlabel('Time (s)')
#%% recorto a ciclos enteros 

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


# obtnego cruces por 0 para obtener frecucuencia
def sinusoide(t,A,f,phi):
    return A*np.sin(2*np.pi*f*t+phi)
frecs=[]
from scipy.optimize import curve_fit
for t,v,f0 in zip([time1,time2,time3,time4,time5,time6],[v1,v2,v3,v4,v5,v6],[1e6,9e5,8e5,7e5,6e5,5.85e5]):
    popt,_ = curve_fit(sinusoide,t,v,p0=[10,f0,0])
    frecs.append(popt[1])
    print(f'{f0:9.1f} Hz | {popt[1]:9.1f} Hz')

#%%
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
import numpy as np
import matplotlib.pyplot as plt

# Paso temporal
dt = t1[1] - t1[0]

# Frecuencia de muestreo
fs = 1/dt

# Cantidad de muestras
N = len(v1r)

# FFT real
V = np.fft.rfft(v1r)

# Eje de frecuencias
freq = np.fft.rfftfreq(N, d=dt)

A = 2*np.abs(V)/N
plt.figure(figsize=(8,4))

plt.plot(freq, A)

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.xlim(0, 2e7)

plt.grid()
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt

dt = t1[1]-t1[0]
fs = 1/dt
N = len(v1r)

# FFT
V = np.fft.rfft(v1r)

# frecuencias
freq = np.fft.rfftfreq(N, dt)/1000

# amplitud pico
A = 2*np.abs(V)/N

# potencia
P = A**2 / 2

# mostrar armónicos importantes
idx = P > 5e-3


for f, a, p in zip(freq[idx], A[idx], P[idx]):
    print(f'{f:8.1f} kHz | A={a:8.4f} | P={p:8.4f}')
    
plt.plot(freq, A)
plt.xlabel('Frecuencia [kHz]')
plt.ylabel('Amplitud')
plt.xlim(0, 2e4)
plt.xticks(freq[idx])
plt.grid()
plt.show()
# %%
