SRC = src
BIN = bin
INCLUDE = include
FLAGS = -lm -lpthread -lX11
CXX = g++
MPI = mpiCC


all: $(BIN)/median-filter $(BIN)/ejemplo

$(BIN)/median-filter: $(SRC)/median-filter.cpp
	$(MPI) $^ $(FLAGS)  -I $(INCLUDE) -o $@

$(BIN)/ejemplo: $(SRC)/ejemplo.cpp
	$(CXX) $^ $(FLAGS) -I $(INCLUDE) -o $@
