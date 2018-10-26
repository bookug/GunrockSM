#compile parameters

CC = g++
NVCC = nvcc -arch=sm_35 -lcudadevrt -rdc=true -G --ptxas-options=-v 
CFLAGS = -c -Wall -O6 -g#-fprofile-arcs -ftest-coverage -coverage #-pg
EXEFLAG = -O6 #-fprofile-arcs -ftest-coverage -coverage #-pg #-O2
	 
#add -lreadline -ltermcap if using readline or objs contain readline
library = -lcudadevrt  #-lgcov -coverage

objdir = ./objs/
objfile = $(objdir)Util.o $(objdir)IO.o $(objdir)Match.o $(objdir)Graph.o

all: run

run: main/run.cpp $(objfile)
	$(NVCC) $(EXEFLAG)  -o run main/run.cpp $(objfile) $(library)

$(objdir)Util.o: util/Util.cpp util/Util.h
	$(CC) $(CFLAGS)  util/Util.cpp -o $(objdir)Util.o

$(objdir)Graph.o: graph/Graph.cpp graph/Graph.h
	$(CC) $(CFLAGS)  graph/Graph.cpp -o $(objdir)Graph.o

$(objdir)IO.o: io/IO.cpp io/IO.h
	$(CC) $(CFLAGS)   io/IO.cpp -o $(objdir)IO.o

$(objdir)Match.o: match/Match.cu match/Match.cuh
	$(NVCC) -c  match/Match.cu -o $(objdir)Match.o 

.PHONY: clean dist tarball test sumlines

clean:
	rm -f $(objdir)*
dist: clean
	rm -f *.txt dig

tarball:
	tar -czvf vf2.tar.gz main util match io graph Makefile README.md objs script vflib2

test: main/test.o $(objfile)
	$(NVCC) -o test main/test.cpp $(objfile) $(library)

sumline:
	bash script/sumline.sh

