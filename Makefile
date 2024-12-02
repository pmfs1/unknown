CCOMP=gcc
NVCOMP=nvcc
ARC=ar
STD_CCOMP_FLAGS=-std=c17 -Wall -pedantic -g -fPIC
CCOMP_FLAGS=$(STD_CCOMP_FLAGS) -fopenmp
CLINK_FLAGS=-Wall -fopenmp
ARC_FLAGS=-rcs

ifdef CUDA_ARCH
	CUDA_ARCH_FLAG=-arch=$(CUDA_ARCH)
else
	CUDA_ARCH_FLAG=-arch=sm_52
endif

MODE=
NVCOMP_FLAGS=--compiler-options '-fPIC' -G $(CUDA_ARCH_FLAG) -I/usr/local/cuda/include
NVLINK_FLAGS=$(CUDA_ARCH_FLAG) -L/usr/local/cuda/lib64
STD_LIBS=-lm
CUDA_STD_LIBS=-lcudart
SRC_DIR=./src
BIN_DIR=./bin
SYSTEM_INCLUDE_DIR=
SYSTEM_LIB_DIR=
OBJS=$(patsubst %.o,$(BIN_DIR)/%.o,$^)
MKDIR=mkdir -p
RM=rm -rf

UNAME_S=$(shell uname -s)
ifeq ($(UNAME_S),Linux)
	SYSTEM_INCLUDE_DIR=/usr/include
	SYSTEM_LIB_DIR=/usr/lib
	STD_LIBS+=-lrt
endif
ifeq ($(UNAME_S),Darwin)
	SYSTEM_INCLUDE_DIR=/usr/local/include
	SYSTEM_LIB_DIR=/usr/local/lib
endif

all: std cuda

install: std-install cuda-install

install-headers:
	sudo $(MKDIR) $(SYSTEM_INCLUDE_DIR)/unknown
	sudo cp $(SRC_DIR)/*.h $(SYSTEM_INCLUDE_DIR)/unknown
	sudo cp $(SRC_DIR)/*.cuh $(SYSTEM_INCLUDE_DIR)/unknown

install-lib:
ifneq ($(MODE), archive)
	sudo cp $(BIN_DIR)/libunknown.so $(SYSTEM_LIB_DIR)
endif
ifeq ($(MODE), archive)
	sudo cp $(BIN_DIR)/libunknown.a $(SYSTEM_LIB_DIR)
endif

std-install: std install-headers install-lib
cuda-install: cuda install-headers install-lib

uninstall: clean
	sudo $(RM) $(SYSTEM_INCLUDE_DIR)/unknown
	sudo $(RM) $(SYSTEM_LIB_DIR)/libunknown.so
	sudo $(RM) $(SYSTEM_LIB_DIR)/libunknown.a

std: create std-build
cuda: create cuda-build

std-build: cortex.o population.o unknown.o
	$(CCOMP) $(CLINK_FLAGS) -shared $(OBJS) $(STD_LIBS) -o $(BIN_DIR)/libunknown.so
	$(ARC) $(ARC_FLAGS) $(BIN_DIR)/libunknown.a $(OBJS)

cuda-build: cortex.o population.o unknown.cuda.o
	$(NVCOMP) $(NVLINK_FLAGS) -shared $(OBJS) $(CUDA_STD_LIBS) -o $(BIN_DIR)/libunknown.so
	$(ARC) $(ARC_FLAGS) $(BIN_DIR)/libunknown.a $(OBJS)

%.o: $(SRC_DIR)/%.c
	$(CCOMP) $(CCOMP_FLAGS) -c $^ -o $(BIN_DIR)/$@

%.cuda.o: $(SRC_DIR)/%.cu
	$(NVCOMP) $(NVCOMP_FLAGS) -c $< -o $(BIN_DIR)/$@

create:
	$(MKDIR) $(BIN_DIR)
	$(MKDIR) $(BIN_DIR)
	bash -c "sudo apt-get update -y && sudo apt install make gcc g++ nvidia-cuda-toolkit -y"

clean:
	$(RM) $(BIN_DIR)
	$(RM) $(BIN_DIR)