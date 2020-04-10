.PHONY = clean

SRC = src
BIN = bin
CC = gcc
NVCC = /usr/local/cuda/bin/nvcc

all: $(BIN) $(BIN)/vectorAddSec

$(BIN):
	mkdir $(BIN)

$(BIN)/vectorAddSec: $(SRC)/vectorAdd.c
	$(CC) -g $^ -o $@

clean:
	rm -rf $(BIN)
