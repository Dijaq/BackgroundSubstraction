all:
	g++ local_svd.cpp -g -o main -std=c++17 -lstdc++fs -larmadillo `pkg-config --cflags --libs opencv`
