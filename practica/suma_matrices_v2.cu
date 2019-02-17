#include <iostream>
#include "svd3_cuda.h"
#include "stdio.h"
#include <chrono>

//#define columnas 640
//#define filas 480
//#define columnas 800
//#define filas 800

#define Threads 16

using namespace std;

__global__ void add(float *m1, float *m3, int filas, int columnas)
{
	int col = blockIdx.x*blockDim.x+threadIdx.x;
	int fil = blockIdx.y*blockDim.y+threadIdx.y;

	int col_min = col-1;
	int col_max = col+1;

	int fil_min = fil-1;
	int fil_max = fil+1;


	if(col < columnas && fil < filas && col_min >= 0 && fil_min >= 0 && fil_max < (filas) && col_max < (columnas))
	{
		int indexi0j0 = (columnas*fil_min)+col_min;
		int indexi0j1 = (columnas*fil_min)+col;
		int indexi0j2 = (columnas*fil_min)+col_max;
		int indexi1j0 = (columnas*fil)+col_min;
		int indexij = (columnas*fil)+col;
		int indexi1j2 = (columnas*fil)+col_max;
		int indexi2j0 = (columnas*fil_max)+col_min;
		int indexi2j1 = (columnas*fil_max)+col;
		int indexi2j2 = (columnas*fil_max)+col_max;

		float u11, u12, u13, u21, u22, u23, u31, u32, u33;
		float s11, s12, s13, s21, s22, s23, s31, s32, s33;
		float v11, v12, v13, v21, v22, v23, v31, v32, v33;

		svd(m1[indexi0j0], m1[indexi0j1], m1[indexi0j2],
        m1[indexi1j0], m1[indexij], m1[indexi1j2],
        m1[indexi2j0], m1[indexi2j1], m1[indexi2j2],
        // output U
        u11, u12, u13, u21, u22, u23, u31, u32, u33,
        // output S
        s11, s12, s13, s21, s22, s23, s31, s32, s33,
        // output V
        v11, v12, v13, v21, v22, v23, v31, v32, v33);
		//m3[index] = m1[index]+m2[index];
		/*m3[indexij] = m1[indexi0j0]+m1[indexi0j1]+m1[indexi0j2]+
			m1[indexi1j0]+m1[indexij]+m1[indexi1j2]+
			m1[indexi2j0]+m1[indexi2j1]+m1[indexi2j2];*/
		m3[indexij] = s11+s22+s33;
	}
}

int main()
{
	int filas = 1080;
	int columnas = 1920;
	int cont = 0;


	cout << "craete" << endl;
	//float a[filas][columnas], c[filas][columnas];
	float a[filas][columnas];
	cout << "craete1" << endl;
	float *dev_a, *dev_b;
	float *dev_c;

	for(int i=0; i<filas; i++)
	{
		//cont = 0;
		for(int j=0; j<columnas; j++)
		{
			//cout << i <<"-" << j<<endl;
			a[i][j] = cont+i;
			cont++;
		}
	}

	cout << "init" << endl;
	auto t11 = std::chrono::high_resolution_clock::now();

	cudaMalloc((void**) &dev_a, filas*columnas*sizeof(float));
	cudaMalloc((void**) &dev_c, filas*columnas*sizeof(float));

	cudaMemcpy(dev_a, a, filas*columnas*sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimThreadsBloque(Threads, Threads);

	float BFloat = (float) columnas / (float) Threads;
  	int B = (int) ceil(BFloat);

  // El grid tendrá B número de bloques en x y y
  	dim3 dimBloques(B, B);

  	add<<<dimBloques, dimThreadsBloque>>>(dev_a, dev_c, filas, columnas);

  	cudaMemcpy(a, dev_c, filas*columnas*sizeof(float), cudaMemcpyDeviceToHost);

  	auto t12 = std::chrono::high_resolution_clock::now();


  	/*for(int i=0; i< filas; i++)
  	{
  		for(int j=0; j<columnas; j++)
  		{
  			cout << i <<"-"<<j<<" : "<<c[i][j] << endl;
  		}
  	}*/

  	cudaFree(dev_a);
  	cudaFree(dev_c);
  	
  	cout << std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t11).count() << endl;


	return 0; 
}
