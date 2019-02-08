#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <armadillo>

#define T 10 // max threads x bloque
#define N 20
#define filas 3
#define columnas 3

using namespace std;

__global__ void sumaMatrices(int *m1, int *m2, int *m3) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int fil = blockIdx.y * blockDim.y + threadIdx.y;

  int indiceij = fil * columnas + col;

  int indicei0j0 = (fil-1) * columnas + (col-1);
  int indicei0j1 = (fil-1) * columnas + (col-1);
  int indicei0j2 = (fil-1) * columnas + (col-1);
  int indicei1j0 = fil * columnas + col;
  int indicei1j2 = fil * columnas + col;
  int indicei2j0 = (fil+1) * columnas + (col+1);
  int indicei2j1 = (fil+1) * columnas + (col+1);
  int indicei2j2 = (fil+1) * columnas + (col+1);


  if (col < columnas && fil < filas) {
  // debido a que en los últimos bloques no se realizan todos los threads
   // m3[indice] = m1[indice] + m2[indice];

    arma::mat m_svd;
    m_svd = {{(double)m1[indicei0j0], (double)m1[indicei0j1], (double)m1[indicei0j2]},
    {(double)m1[indicei1j0], (double)m1[indiceij], (double)m1[indicei1j2]},
    {(double)m1[indicei2j0], (double)m1[indicei2j1], (double)m1[indicei2j2]}};

    arma::mat U2, V2;
    arma::vec w2;
    
    arma::svd(U2, w2, V2, m_svd);
    m3[indiceij] = (w2[2]+w2[1])/w2[0];
  }
}

int main(int argc, char** argv) {

  int m1[filas][columnas];
  int m2[filas][columnas];
  int m3[filas][columnas];

  /*int **m1 = new int*[filas];
  int **m2 = new int*[filas];
  int **m3 = new int*[filas];

  for(int i=0; i<filas; i++)
  {
    m1[i] = new int[columnas];
    m2[i] = new int[columnas];
    m3[i] = new int[columnas];
  }*/
  int i, j;
  int c = 0;

  /* inicializando variables con datos foo*/
  for (i = 0; i < filas; i++) {
    c = 0;
    for (j = 0; j < columnas; j++) {
      m1[i][j] = c;
      m2[i][j] = c;
      c++;
    }
  }

  int *dm1, *dm2, *dm3;

  cudaMalloc((void**) &dm1, filas * columnas * sizeof(int**));
  cudaMalloc((void**) &dm2, filas * columnas * sizeof(int**));
  cudaMalloc((void**) &dm3, filas * columnas * sizeof(int**));

  // copiando memoria a la GPGPU
  cudaMemcpy(dm1, m1, filas * columnas * sizeof(int**), cudaMemcpyHostToDevice);
  cudaMemcpy(dm2, m2, filas * columnas * sizeof(int**), cudaMemcpyHostToDevice);

  // cada bloque en dimensión x y y tendrá un tamaño de T Threads
  dim3 dimThreadsBloque(T, T);

  // Calculando el número de bloques en 1D
  float BFloat = (float) columnas / (float) T;
  int B = (int) ceil(BFloat);

  // El grid tendrá B número de bloques en x y y
  dim3 dimBloques(B, B);

  // Llamando a ejecutar el kernel
  sumaMatrices<<<dimBloques, dimThreadsBloque>>>(dm1, dm2, dm3);

  // copiando el resultado a la memoria Host
  cudaMemcpy(m3, dm3, filas * columnas * sizeof(int**), cudaMemcpyDeviceToHost);
  //cudaMemcpy(m2, dm2, N * N * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(dm1);
  cudaFree(dm2);
  cudaFree(dm3);

  printf("\n");

  for (i = 0; i < filas; i++) {
    for (j = 0; j < columnas; j++) {
      printf(" [%d,%d]=%d", i, j, m3[i][j]);

    }
    printf("\n\n");

  }
  
  cout << "B: "<<B << endl;
  cout << "DimBloques.X: " << dimBloques.x << " DimbBloques.Y: " << dimBloques.y << endl;
  cout << "DimThreadsBloque.X: " << dimThreadsBloque.x << " DimThreadsBloque.Y " << dimThreadsBloque.y <<endl;

  return (EXIT_SUCCESS);
}