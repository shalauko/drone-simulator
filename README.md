# drone-simulator

Dependencies
- numpy
- scipy
- PIL
- matplotlib
- sys
- time
- csv
- control

Almoust all of these packages are preinstalled in Anaconda 1.9.6. Just install anaconda for Python 3 -> https://www.anaconda.com/

Additionally, you have to install only "control" package in version 0.8.1 at least -> https://python-control.readthedocs.io/en/0.8.1/

average RMS of position of 10 simulations each case in python (step time 0.01 sec)
for 50 sec: 0,0276 (vor median velocity ~0.5)
for 20 sec: 0,0679 (vor median velocity ~1.3)
for 10 sec: 0,1721 (vor median velocity ~3.2)

Conclusion -> With increasing of median velocity increase error of position (becouse of inertia) 
Our controller works quite good with speed lower than 1.5 m/s (54 km/h), what is really big for drones and appears in the usage really rarely.
