𝓾𝓷𝓴𝓷𝓸𝔀𝓷 _is a spiking neural network library leveraging grid layouts and kernels to enhance efficiency in highly parallel environments and striving to closely mimic the biological brain without compromising performance. By being constantly influenced by its inputs, the network leads to potentially unexpected (emergent) outcomes._

> [!NOTE]
> _Before installing the library please make sure you have all of the required dependencies installed by running the following command: `sudo apt-get update -y && sudo apt install software-properties-common make gcc g++ opencl-headers ocl-icd-opencl-dev libopencv-videoio-dev libopencv-core-dev nvidia-cuda-toolkit -y`._
> - _To install the default (CPU) package in a system-wide dynamic or static library, use: `make install` or `make std-install`._
> - _To install the CUDA parallel (GPU) package in a system-wide dynamic or static library, use: `make cuda-install`._
>   - _Optionally, you can specify the compute capability of your GPU, using `make cuda-install CUDA_ARCH=sm_61`, which allows for some extra optimizations._

<!-- make uninstall && sudo apt-get update -y && sudo apt install make gcc g++ nvidia-cuda-toolkit -y && make install && make cuda-install CUDA_ARCH=sm_61 && cd .github/workflows/_benchmark/cuda && make && ./bin/bench && cd .. && cd std && make && ./bin/bench && cd ../../../.. && make uninstall -->