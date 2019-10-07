# Hodographic Shaping in Python
This repository containts an implementation of the time-driven hodographic shaping method [1] for the computation of interplanetary, low-thrust spacecraft trajectories in Python. It provides an efficient way to find good, sub-optimal trajectories in preliminary optimization, including searching the launch window, optimizing transfers using additional base functions, and linking multiple transfers to create realistic mission scenarios. Functions to apply impulsive shots, perform flybys and rendezvous as well as extended plotting functionality complement its usefulness for preliminary global trajectory optimization.

Documentation is provided in the shape of Jupyter Notebooks which show the core functionality in an interactive fashion. Exported Pdfs can be found in the 'tutorials_pdf' folder. The notebooks and code is best run using a conda environment and requires the following packages:  
```pykep pygmo nlopt scipy numpy matplotlib```  
In order to use high precision ephemeris data the NASA spice kernel [de430.bsp](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/) (114 Mb) needs to be copied into the 'ephemerides' folder. It includes the positions and velocities of the Sun, the planets, and the moon over a wide time span. The kernels for Vesta and Ceres are already included. Kernel files for other small bodies or dates can be retrieved from NASA's [Horizons system](https://ssd.jpl.nasa.gov/x/spk.html).

The code is the result of my MSc thesis work at TU Delft [2], where I combined a Genetic Algorithm with a Neural Network surrogate model in order to improve the optimization of interplanetary, linked, low-thrust trajectories. If there is interest in the code for those surrogate assisted optimizations please let me know.

Example reproducing the trajectory of the Dawn mission and the corresponding thrust acceleration profile. The spacecraft leaves Earth with an impulsive shot, performs a flyby at Mars, rendezvous at Vesta, stays for a while, and finally transfers to Ceres.
<p align="center">
  <img width="720" src="images/dawn.svg?sanitize=true">
</p>
<p align="center">
  <img width="720" src="images/dawn-thrust.svg?sanitize=true">
</p>

[1] D. Gondelach and R. Noomen, [Hodographic-shaping method for low-thrust interplanetary trajectory design](https://arc.aiaa.org/doi/abs/10.2514/1.A32991), Journal of Spacecraft and Rockets, 2015  
[2] L. Stubbig, [Investigating the use of neural network surrogate models in the evolutionary optimization of interplanetary low-thrust trajectories](http://resolver.tudelft.nl/uuid:97481b94-dcd8-4f5d-bd95-e91f3146f69a), MSc Thesis, TU Delft, 2019
