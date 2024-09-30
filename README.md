_A spiking neural network library leveraging grid layouts and kernels to enhance efficiency in highly parallel environments._
  - _This implementation strives to closely mimic the biological brain without compromising performance._
  - _The learning process is continuous, with no clear separation between training, validation, and deployment._
  - _The network is constantly influenced by its inputs, leading to potentially unexpected (emergent) outcomes._

> [!TIP]
> _All the following commands install `pmfs1/unknown` as a dynamic library by default, but you can tell the `make` command to install it as a static library by setting the dedicated variable: `make install MODE=archive`._

> [!IMPORTANT]  
> - _Run `make install` or `make std-install` to install the default (CPU) package in a system-wide dynamic or static library._
> - _Run `make cuda-install` to install the CUDA parallel (GPU) package in a system-wide dynamic or static library._
>   - _Optionally, you can specify the compute capability of your GPU with the dedicated variable `CUDA_ARCH` which allows for some extra optimizations: `make cuda-install CUDA_ARCH=sm_61`._
> 
> _Before installing the library please make sure you have all of the required dependencies installed by running the following command: `sudo apt-get update -y && sudo apt install software-properties-common make gcc g++ opencl-headers ocl-icd-opencl-dev libopencv-videoio-dev libopencv-core-dev nvidia-cuda-toolkit -y`._

<!-- 
```bash
make uninstall && sudo apt-get update -y && sudo apt install software-properties-common make gcc g++ opencl-headers ocl-icd-opencl-dev libopencv-videoio-dev libopencv-core-dev nvidia-cuda-toolkit -y && make install && make cuda-install CUDA_ARCH=sm_61 && cd benchmark/cuda && make && ./bin/bench && cd .. && cd std && make && ./bin/bench && cd ../..
```
 -->