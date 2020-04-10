.PHONY = clean

SRC = src
BIN = bin

CC = gcc
NVCC = /usr/local/cuda/bin/nvcc

FLAGS = -lm -fopenmp
MATH = -lm

all: $(BIN) $(BIN)/vectorAddSec

$(BIN):
	mkdir $(BIN)

$(BIN)/vectorAddSec: $(SRC)/vectorAdd.c
	$(CC) $^ -o $@ $(FLAGS)

clean:
	rm -rf $(BIN)
