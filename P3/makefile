SRC = src
BIN = bin
INCLUDE = include
FLAGS = -lm -lpthread -lX11
CXX = g++


all: $(BIN)/median-filter $(BIN)/ejemplo

$(BIN)/median-filter: $(SRC)/median-filter.cpp
	$(CXX) $^ $(FLAGS) -I $(INCLUDE) -o $@

$(BIN)/ejemplo: $(SRC)/ejemplo.cpp
	$(CXX) $^ $(FLAGS) -I $(INCLUDE) -o $@
