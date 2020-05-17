.PHONY = clean

SRC = src
BIN = bin
OBJ = obj
INCLUDE = include
OBJECTS = $(OBJ)/image_utils.o $(OBJ)/main.o $(OBJ)/medianKernel.o

NVCC = /usr/local/cuda/bin/nvcc
CC = g++

FLAGS = -L/usr/local/cuda/lib64 -lcudart -lcuda -lm -lpthread -lX11
NFLAGS = -m64 --std=c++11
OPT = -O2


all: $(BIN) $(OBJ) $(BIN)/medianFilter 

$(BIN):
	mkdir $(BIN)

$(OBJ):
	mkdir $(OBJ)

$(BIN)/medianFilter: $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(FLAGS)

$(OBJ)/image_utils.o: $(SRC)/image_utils.cpp
	$(CC) -c $<  -I $(INCLUDE) $(OPT) -o $@

$(OBJ)/main.o: $(SRC)/main.cpp
	$(CC) -c $< -I $(INCLUDE) $(OPT) -o $@

$(OBJ)/medianKernel.o: $(SRC)/medianKernel.cu
	$(NVCC) $(NFLAGS) -c $< -I $(INCLUDE) -o $@

clean:
	rm -rf $(BIN) $(OBJ)
