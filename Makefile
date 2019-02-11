all:
	g++ local_svd.cpp -fopenmp -o main -std=c++11 -lpthread -lstdc++fs  -DWITH_CUDA=ON -larmadillo `pkg-config --cflags --libs opencv`
exec:
	g++ local_svd_copia.cpp -fopenmp -o main -std=c++11 -lstdc++fs  -DWITH_CUDA=ON -larmadillo `pkg-config --cflags --libs opencv`



