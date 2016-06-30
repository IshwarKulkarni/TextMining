#Make file for TextMining project

VALGRIND ?= "false"

CPP      = g++ -std=c++11 -pedantic
CPPFLAGS = -Wall -g -I../../src -DDATA_LOCATION="\"../../data/\"" -lpthread

SOURCES	+= $(shell find ../../src/utils/ -name *.cxx)
SOURCES	+= $(shell find ../../src/textmining -name *.cxx)

ifeq "$(VALGRIND)" "true"
CPPFLAGS += -DVALGRIND
endif

VALGRINDEXE = valgrind --leak-check=full  --track-origins=yes --show-leak-kinds=all 
VALGRINDOPT = ""

OBJECTS    := ${SOURCES:.cxx=.o}

release: 	CPPFLAGS += -DNDEBUG -O3 
debug: 		CPPFLAGS += -D_DEBUG -Og -O0 -ggdb -DGLIBCXX_FORCE_NEW

release: 	all
debug: 		all
TextMining: 	all

all:
	@if $(CPP) $(SOURCES) $(CPPFLAGS) -o TextMining ; then \
	echo "Build completed, binary: TextMining" ; \
    if [ "$(VALGRIND)" = "true" ] ; then \
    echo "Running Valgrind; Check file \".valgrind\" for results" ; \
    $(VALGRINDEXE) ./TextMining $(VALGRINDOPT)  >& .valgrind; \
    fi  \
    else \
	echo "Build Failed!" ; fi

clean:
	rm TextMining
