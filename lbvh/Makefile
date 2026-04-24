CC = gcc
CFLAGS = -O2 -fPIC -Wall
LDFLAGS = -shared -lOpenCL -ldl
INCLUDE = -Iinclude

liblbvh.so: lbvh.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(INCLUDE)


test:
	$(CC) -O2 -o test test_simple.c lbvh.c -lOpenCL -ldl $(INCLUDE)

clean:
	rm -f liblbvh.so
	rm -f test