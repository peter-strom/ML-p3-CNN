CC=g++
CFLAGS=-Wall
OBJS=*.cpp
OUTPUT=-o main
LIBRARY= -lgpiodcxx

all: make run

make:
	$(CC) $(OBJS) $(OUTPUT) $(CFLAGS) $(LIBRARY)  
run :
	./main

