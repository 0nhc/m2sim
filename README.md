# $M^2Sim$

## 1. Prerequisites
### 1.1 Deep Learning Configurations
For example, my configurations are:
* RTX 3090 Ti
* NVIDIA Driver 535.154.05
* CUDA 12.2
* cuDNN 8.9.4.25
* torch==2.2.0
* tensorflow==2.15.0

## 2. Installation
```sh
git clone https://github.com/0nhc/m2sim.git
cd m2sim/src/ && cython -a utils_cython.pyx && python setup.py build_ext --inplace && cd ..
```