CC=g++
CFLAGS= -std=c++11 -c -g -Wall
SOURCE=main.cpp ann.cpp
LDFLAGS=
OBJECTS= $(SOURCE:.cpp=.o)

EXECUTABLE= ann


all:  $(SOURCE) $(EXECUTABLE)


$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@


clean: 
	rm -f *.o $(EXECUTABLE)

