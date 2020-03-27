.PHONY = clean
SRC = src
BIN = bin
INCLUDE = include
FLAGS = -lm -lpthread -lX11 -O2
JPEG = -ljpeg
FOMP = -fopenmp
CXX = g++
MPI = mpiCC


all: $(BIN) $(BIN)/median-filter $(BIN)/median-filter-sec

$(BIN):
	mkdir $(BIN)

$(BIN)/median-filter: $(SRC)/median-filter.cpp
	$(MPI) $^ $(FLAGS) $(FOMP) -I $(INCLUDE) -o $@

$(BIN)/median-filter-sec: $(SRC)/median-filter-sec.cpp
	$(CXX) $^ $(FLAGS) -I $(INCLUDE) -o $@

clean:
	rm -rf $(BIN)
