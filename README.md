_A spiking neural network library inspired by cellular automata; it borrows concepts such as grid layout and kernels from cellular automata to boost efficiency in highly parallel environments. The implementation aims at mimicking a biological brain as closely as possible without losing performance. The learning pattern of one of these neural networks is continuos, with no distinction between training, validation and deploy. The network is continuously changed by its inputs and therefore can produce unexpected (emerging) results._

<!-- ## Shared library installation (Linux)
All the following commands install unknown as a dynamic library by default, but you can tell the make command to install it as a static library by setting the dedicated variable `MODE=archive`:<br/>
`make install MODE=archive`<br/>

You can also specify the C compiler by setting the dedicated variable `CCOMP=gcc-14`:<br/>
`make install CCOMP=gcc-14`

### Standard
Run `make install` or `make std-install` to install the default (CPU) package in a system-wide dynamic or static library.<br/>

### CUDA
Run `make cuda-install` to install the CUDA parallel (GPU) package in a system-wide dynamic or static library.<br/>

Optionally you can specify the compute capability of your GPU with the dedicated variable `CUDA_ARCH`. This allows for some extra optimizations:<br/>
`make cuda-install CUDA_ARCH=sm_61`<br/>

Warnings:<br/>
* The CUDA version only works with NVIDIA GPUS<br/>
* The CUDA version requires the CUDA SDK and APIs to work<br/>
* The CUDA SDK or APIs are not included in any install_deps.sh script<br/>

### OpenCL
TODO

### Uninstall
Run `make uninstall` to uninstall any previous installation.

WARNING: Every time you `make` a new package the previous installation is overwritten.

## How to use
### Header files
Once the installation is complete you can include the library by `#include <unknown/unknown.h>` and directly use every function in the packages you compiled.<br/>

### Linking
During linking you can specify `-lunknown` in order to link the compiled functions. -->
