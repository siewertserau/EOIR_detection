INCLUDE_DIRS =  -I/usr/include
LIB_DIRS = 
CC=g++

CDEFS=
CFLAGS= -O0 -g $(INCLUDE_DIRS) $(CDEFS)
LIBS= -lrt
CPPLIBS= -L /usr/lib/ -lopencv_core -lopencv_flann -lopencv_video -lopencv_contrib

HFILES= slic.h
CFILES= 
CPPFILES= final.cpp slic.cpp

SRCS= ${HFILES} ${CFILES}
CPPOBJS= ${CPPFILES:.cpp=.o}

all:	final 

clean:
	-rm -f *.o *.d
	-rm -f final

distclean:
	-rm -f *.o *.d

final: final.o
	$(CC) $(LDFLAGS) $(CFLAGS) ${CPPFILES} -ggdb `pkg-config --cflags opencv` -o $@ `pkg-config --libs opencv` $(CPPLIBS)

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<

.cpp.o:
	$(CC) $(CFLAGS) -c $<
