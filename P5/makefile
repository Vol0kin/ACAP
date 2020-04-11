.PHONY = clean

SRC = src
BIN = bin

CC = gcc
NVCC = /usr/local/cuda/bin/nvcc

FLAGS = -lm -fopenmp
FCUDA = -m64 -lm -Xcompiler -fopenmp

all: $(BIN) $(BIN)/vectorAddSec $(BIN)/vectorAddCUDA $(BIN)/deviceProperties

$(BIN):
	mkdir $(BIN)

$(BIN)/vectorAddSec: $(SRC)/vectorAdd.c
	$(CC) $^ -o $@ $(FLAGS)

$(BIN)/deviceProperties: $(SRC)/deviceProperties.cu
	$(NVCC) $^ -o $@

$(BIN)/vectorAddCUDA: $(SRC)/vectorAdd.cu
	$(NVCC) $^ -o $@ $(FCUDA)

clean:
	rm -rf $(BIN)
