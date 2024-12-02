CCOMP=gcc
STD_CCOMP_FLAGS=-std=c17 -Wall -pedantic -g
CCOMP_FLAGS=$(STD_CCOMP_FLAGS)
CLINK_FLAGS=-Wall
STD_LIBS=-lm
GRAPHICS_LIBS=-lsfml-graphics -lsfml-window -lsfml-system
SPIKES_COUNT_LIBS=-lncurses
TIME_CUDA_LIBS=-lcuda -lcudart -lcublas -lcurand
UNKNOWN_LIBS=-lunknown
SRC_DIR=./src
BIN_DIR=./bin
OUT_DIR=./out
OBJS=$(patsubst %.o,$(BIN_DIR)/%.o,$@.o)
MKDIR=mkdir -p
RM=rm -rf
UNAME_S=$(shell uname -s)

ifeq ($(UNAME_S),Linux)
	STD_LIBS+=-lrt
endif
ifeq ($(UNAME_S),Darwin)
	CCOMP_FLAGS+=-I/usr/local/include
	CLINK_FLAGS+=-L/usr/local/lib
endif

all: clean bench

bench: create utils.o
	$(CCOMP) $(CCOMP_FLAGS) -c $(SRC_DIR)/$@.c -o $(BIN_DIR)/$@.o
	$(CCOMP) $(CLINK_FLAGS) $(BIN_DIR)/bench.o $(BIN_DIR)/utils.o -o $(BIN_DIR)/$@ $(STD_LIBS) $(UNKNOWN_LIBS)

utils.o: create
	$(CCOMP) $(CCOMP_FLAGS) -c $(SRC_DIR)/utils.c -o $(BIN_DIR)/utils.o

create:
	$(MKDIR) $(BIN_DIR)
	$(MKDIR) $(BIN_DIR)
	$(MKDIR) $(OUT_DIR)
	bash -c "sudo apt-get update -y && sudo apt install make gcc g++ libopencv-videoio-dev libopencv-core-dev -y"

clean:
	$(RM) $(BIN_DIR)
	$(RM) $(BIN_DIR)
	$(RM) $(OUT_DIR)
