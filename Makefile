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
	CUDA_ARCH_FLAG=
endif

MODE=

NVCOMP_FLAGS=--compiler-options '-fPIC' -G $(CUDA_ARCH_FLAG)
NVLINK_FLAGS=$(CUDA_ARCH_FLAG)

STD_LIBS=-lm
CUDA_STD_LIBS=-lcudart

SRC_DIR=./src
BLD_DIR=./bld
BIN_DIR=./bin

SYSTEM_INCLUDE_DIR=
SYSTEM_LIB_DIR=

OBJS=$(patsubst %.o,$(BLD_DIR)/%.o,$^)

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

all: std

install: std-install

install-headers:
	@printf "\nInstalling headers...\n\n"
	sudo $(MKDIR) $(SYSTEM_INCLUDE_DIR)/unknown
	sudo cp $(SRC_DIR)/*.h $(SYSTEM_INCLUDE_DIR)/unknown

install-lib:
ifneq ($(MODE), archive)
	@printf "\nInstalling dynamic library...\n\n"
	sudo cp $(BLD_DIR)/libunknown.so $(SYSTEM_LIB_DIR)
endif
ifeq ($(MODE), archive)
	@printf "\nInstalling static library...\n\n"
	sudo cp $(BLD_DIR)/libunknown.a $(SYSTEM_LIB_DIR)
endif

std-install: std install-headers install-lib
	@printf "\nInstallation complete!\n\n"

cuda-install: cuda install-headers install-lib
	@printf "\nInstallation complete!\n\n"

uninstall: clean
	sudo $(RM) $(SYSTEM_INCLUDE_DIR)/unknown
	sudo $(RM) $(SYSTEM_LIB_DIR)/libunknown.so
	sudo $(RM) $(SYSTEM_LIB_DIR)/libunknown.a
	@printf "\nSuccessfully uninstalled.\n\n"

std: create std-build
cuda: create cuda-build

std-build: cortex.o utils.o population.o unknown_std.o
	$(CCOMP) $(CLINK_FLAGS) -shared $(OBJS) $(STD_LIBS) -o $(BLD_DIR)/libunknown.so
	$(ARC) $(ARC_FLAGS) $(BLD_DIR)/libunknown.a $(OBJS)
	@printf "\nCompiled $@!\n"

cuda-build: cortex.o utils.o population.o unknown_cuda.o
	$(NVCOMP) $(NVLINK_FLAGS) -shared $(OBJS) $(CUDA_STD_LIBS) -o $(BLD_DIR)/libunknown.so
	$(ARC) $(ARC_FLAGS) $(BLD_DIR)/libunknown.a $(OBJS)
	@printf "\nCompiled $@!\n"

%.o: $(SRC_DIR)/%.c
	$(CCOMP) $(CCOMP_FLAGS) -c $^ -o $(BLD_DIR)/$@

%.o: $(SRC_DIR)/%.cu
	$(NVCOMP) $(NVCOMP_FLAGS) -c $^ -o $(BLD_DIR)/$@


create:
	$(MKDIR) $(BLD_DIR)
	$(MKDIR) $(BIN_DIR)

clean:
	$(RM) $(BLD_DIR)
	$(RM) $(BIN_DIR)
