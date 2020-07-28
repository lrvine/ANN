CC = g++
CFLAGS = -std=c++2a -c -Ofast -march=native -Wall
SOURCE = main.cc machinelearning.cc ann.cc
LDFLAGS =
OBJECTS = $(SOURCE:.cc=.o)

EXECUTABLE = ann


all:  $(SOURCE) $(EXECUTABLE)


$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cc.o:
	$(CC) $(CFLAGS) $< -o $@


clean: 
	rm -f *.o $(EXECUTABLE)

