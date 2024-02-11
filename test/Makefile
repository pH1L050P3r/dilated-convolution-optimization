CC = g++
CFLAGS = -Wall -Wextra -pedantic -std=c++11

all: dilated_convolution

dilated_convolution: test_dilated_convolution.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f dilated_convolution