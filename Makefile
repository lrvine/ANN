CC=g++
#CFLAGS= -std=c++11 -c -g -Wall
#CFLAGS= -c -Ofast -march=native -mavx2 -fslp-vectorize-aggressive -Rpass-analysis=loop-vectorize -Wall
CFLAGS= -c -Ofast -march=native -mavx2 -fslp-vectorize-aggressive -Wall
SOURCE=main.cc ann.cc
LDFLAGS=
OBJECTS= $(SOURCE:.cc=.o)

EXECUTABLE= ann


all:  $(SOURCE) $(EXECUTABLE)


$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cc.o:
	$(CC) $(CFLAGS) $< -o $@


clean: 
	rm -f *.o $(EXECUTABLE)

