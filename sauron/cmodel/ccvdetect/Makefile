
CCV_PATH=/datas/pkgs/ccv-unstable/lib
FLAGS=-fPIC

all: ccvdetect.c
	gcc -o ccvdetect ccvdetect.c -I/usr/include/python2.7 -L$(CCV_PATH) \
		-I$(CCV_PATH) -lccv `cat $(CCV_PATH)/.deps` 
#	gcc -o t.so -shared ccvdetect.c -I/usr/include/python2.7 -L$(CCV_PATH) \
#		-I$(CCV_PATH) -lccv `cat $(CCV_PATH)/.deps` $(FLAGS)
#	swig -python ccvdetect.i
#	gcc -c ccvdetect.c ccvdetect_wrap.c -I/usr/include/python2.7 -L$(CCV_PATH) \
#		-I$(CCV_PATH) -lccv `cat $(CCV_PATH)/.deps` $(FLAGS)
#	ld $(FLAGS) -shared ccvdetect.o ccvdetect_wrap.o $(CCV_PATH)/libccv.a -o _ccvdetect.so 

clean:
	rm -f *.o *.so ccvdetect
