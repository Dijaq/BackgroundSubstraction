# Background Substraction using Local SVD Binary Pattern
El presente proyecto es una implementación en c++ del algoritmo LSBP

# Ejecucion
Requirements:
- Sistema Operativo: Ubuntu
- Opencv 3.2.0 o superior
- Compilador gcc, g++

To run the code:
- Clone o download the source code.
- Open a terminal in the repository folder.
- Compile with the command:
-using threads
g++ local_svd.cpp -fopenmp -o main -std=c++11 -lpthread -lstdc++fs  -DWITH_CUDA=ON -larmadillo `pkg-config --cflags --libs opencv`
- using cuda
nvcc local_svd.cu -o main2 -std=c++11 -lpthread  -larmadillo `pkg-config --cflags --libs opencv`

-using threads
or write "make"
- Run with the command:
./main
- using cuda
or write "make exec"
- Run with the command:
./main2

# 1. Results
## 1.1 Time comparation
![Alt text](https://github.com/Dijaq/BackgroundSubstraction/tree/master/datos/Table.PNG?raw=true "Title")



# Fuentes
- [1] Lili Guo, Dan Xu, and Zhenping Qiang, “Background Subtraction using Local SVD Binary Pattern”.
- [2] M. Hofmann, P. Tiefenbacher, and G. Rigoll, “Background segmentation with feedback: The pixel-based adaptive segmenter.”.
