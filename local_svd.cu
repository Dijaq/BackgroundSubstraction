#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <list>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <armadillo>
#include <chrono>


#define Threads 10 // max threads x bloque
#define N 20
#define filas 10
#define columnas 20

using namespace std;
using namespace cv;

void extract_LSBP(Mat frame, Mat &output, int tau);
Mat SVD_init(Mat frame, int samples);
Mat SVD_step(Mat, int, int, int, double, double);
int clip(int i, int inferior, int superior, int val_range);
int Hamming_distance(Mat svd_frame, Mat svd_sample, int i, int j, double tau);
double _SVD(arma::mat matriz);

Mat D, fr, lsbp;
list<Mat> samples_lsbp;
list<Mat> samples_frame;
int heigth, width;
Mat R, T;

__global__ void sumaMatrices(Mat *m1, Mat *m2, Mat *m3) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int fil = blockIdx.y * blockDim.y + threadIdx.y;

  int indice = fil * columnas + col;


  if (col < columnas && fil < filas) {
  // debido a que en los últimos bloques no se realizan todos los threads
    //m3[indice] = m1[indice] + m2[indice];
    m3->data[indice] = m1->data[indice] + m2->data[indice];
  }
}

int main(int argc, char** argv) {

//  int m1[filas][columnas];
//  int m2[filas][columnas];
//  int m3[filas][columnas];
  Mat m1 = Mat::ones(filas, columnas, CV_32FC1);
  Mat m2 = Mat::ones(filas, columnas, CV_32FC1);
  Mat m3 = Mat::zeros(filas, columnas, CV_32FC1);

//  cuda::GpuMat d_m1.upload(m1);
//  cuda::GpuMat d_m2;
//  cuda::GpuMat d_m3;

  int i, j;
  int c = 0;

  /* inicializando variables con datos foo*/
  for (i = 0; i < filas; i++) {
    c = 0;
    for (j = 0; j < columnas; j++) {
      //m1[i][j] = c;
      //m2[i][j] = c;
      m1.at<float>(i,j) = c;
      m2.at<float>(i,j) = c;
      c++;
    }
  }

  //int *dm1, *dm2, *dm3;
  Mat *dm1, *dm2, *dm3;

  const size_t size_m1 = sizeof(m1);

  cudaMalloc((void**) &dm1, size_m1);
  cudaMalloc((void**) &dm2, size_m1);
  cudaMalloc((void**) &dm3, size_m1);

  // copiando memoria a la GPGPU
  cudaMemcpy(dm1, &m1, size_m1, cudaMemcpyHostToDevice);
  cudaMemcpy(dm2, &m2, size_m1, cudaMemcpyHostToDevice);

  // cada bloque en dimensión x y y tendrá un tamaño de T Threads
  dim3 dimThreadsBloque(Threads, Threads);

  // Calculando el número de bloques en 1D
  float BFloat = (float) columnas / (float) Threads;
  int B = (int) ceil(BFloat);

  // El grid tendrá B número de bloques en x y y
  dim3 dimBloques(B, B);

  // Llamando a ejecutar el kernel
  sumaMatrices<<<dimBloques, dimThreadsBloque>>>(dm1, dm2, dm3);

  // copiando el resultado a la memoria Host
  cudaMemcpy(&m3, dm3, size_m1, cudaMemcpyDeviceToHost);
  //cudaMemcpy(m2, dm2, N * N * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(dm1);
  cudaFree(dm2);
  cudaFree(dm3);

  printf("\n");

  for (i = 0; i < filas; i++) {
    for (j = 0; j < columnas; j++) {
      //printf(" [%d,%d]=%d", i, j, m3[i][j]);
      cout << "i, j: " << i <<" - " << j <<" : " << m3.at<float>(i,j) << endl;

    }
    printf("\n\n");

  }
  
  cout << "B: "<<B << endl;
  cout << "DimBloques.X: " << dimBloques.x << " DimbBloques.Y: " << dimBloques.y << endl;
  cout << "DimThreadsBloque.X: " << dimThreadsBloque.x << " DimThreadsBloque.Y " << dimThreadsBloque.y <<endl;

  return (EXIT_SUCCESS);
}

Mat SVD_init(Mat frame, int samples)
{
  Mat svd = Mat::zeros(width+2, heigth+2, CV_32FC1);
  
  extract_LSBP(frame, svd, 0.05);

  samples_lsbp.push_back(svd);
  //cout << "Impr2" << endl;
  samples_frame.push_back(frame);
  //cout << "Impr3" << endl;
  int i0, j0;

  for(int k=1; k<samples;k++)
  {
    lsbp = svd.clone();
    fr = frame.clone();

    for(int i=0; i<frame.rows; i++)
    {
      for(int j=0; j<frame.cols; j++)
      {
        i0 = clip(i,10,frame.rows-10,10);
        j0 = clip(j,10,frame.cols-10,10);

        fr.at<Vec3b>(i0,j0) = frame.at<Vec3b>(i,j);
        lsbp.at<double>(i0,j0) = svd.at<double>(i, j); 
      }
    }

    samples_lsbp.push_back(lsbp);
    samples_frame.push_back(fr);

    lsbp.release();
    fr.release();
  }

  return frame;
}

void extract_LSBP(Mat frame, Mat &r_lsbp, int tau=0.05)
{
  Mat intensity;
  cvtColor(frame, intensity, COLOR_BGR2GRAY);
  Mat intensity_fr = Mat::zeros(frame.rows+2, frame.cols+2, CV_8UC1);

  for(int i=0; i<(frame.rows+2)-1; i++)
  {
    for(int j=0; j<(frame.cols+2)-1; j++)
    {
      //((Scalar)intensity_fr.at<uchar>(i+1,j+1))[0] = ((Scalar)intensity.at<uchar>(i,j))[0]; 
      intensity_fr.at<uchar>(i+1,j+1) = intensity.at<uchar>(i,j); 
      //cout << "::::> " <<i<<" - " <<j<<": "<<((Scalar)intensity_fr.at<uchar>(i+1,j+1))[0] << "--"<< ((Scalar)intensity.at<uchar>(i,j))[0]<<endl;
      //waitKey(10);
    }
  }

  /*cout << "LSBP" << endl;
  imshow("imagen", intensity_fr);
  waitKey(5000);*/
  


//Falta asignar los bordes

  //Mat m_svd = Mat::zeros(3, 3, CV_8UC1);
  
  //Mat LSBP = Mat::zeros(width, heigth, CV_8UC9);
  
//SVD descomposicion
  auto t11 = std::chrono::high_resolution_clock::now();
  for(int i=1; i<intensity_fr.rows-1; i++)
  {
    for(int j=1; j<intensity_fr.cols-1; j++)
    {
      //cout << ((Scalar)intensity.at<uchar>(i-1,j-1))[0] << endl;
      //cout << ((Scalar)intensity.at<uchar>(i-1,j))[0] << endl;
      //cout << ((Scalar)intensity.at<uchar>(i-1,j+1))[0] << endl;
      //Primera fila
      /*m_svd.at<uchar>(0,0) = ((Scalar)intensity.at<uchar>(i-1,j-1))[0];
      m_svd.at<uchar>(0,1) = ((Scalar)intensity.at<uchar>(i-1,j))[0];
      m_svd.at<uchar>(0,2) = ((Scalar)intensity.at<uchar>(i-1,j+1))[0];

      //Segunda fila
      m_svd.at<uchar>(1,0) = ((Scalar)intensity.at<uchar>(i,j-1))[0];
      m_svd.at<uchar>(1,1) = ((Scalar)intensity.at<uchar>(i,j))[0];
      m_svd.at<uchar>(1,2) = ((Scalar)intensity.at<uchar>(i,j+1))[0];

      //Tercera fila
      m_svd.at<uchar>(2,0) = ((Scalar)intensity.at<uchar>(i+1,j-1))[0];
      m_svd.at<uchar>(2,1) = ((Scalar)intensity.at<uchar>(i+1,j))[0];
      m_svd.at<uchar>(2,2) = ((Scalar)intensity.at<uchar>(i+1,j+1))[0];*/

      arma::mat m_svd;
      m_svd = {{((Scalar)intensity_fr.at<uchar>(i-1,j-1))[0],
      ((Scalar)intensity_fr.at<uchar>(i-1,j))[0],
      ((Scalar)intensity_fr.at<uchar>(i-1,j+1))[0]},

      {((Scalar)intensity_fr.at<uchar>(i,j-1))[0],
      ((Scalar)intensity_fr.at<uchar>(i,j))[0],
      ((Scalar)intensity_fr.at<uchar>(i,j+1))[0]},

      {((Scalar)intensity_fr.at<uchar>(i+1,j-1))[0],
      ((Scalar)intensity_fr.at<uchar>(i+1,j))[0],
      ((Scalar)intensity_fr.at<uchar>(i+1,j+1))[0]}};

      /*cout << "AAAAAAA: " <<endl;
      cout << m_svd << endl;
      waitKey(5000);*/

      //cout << "Double: " << i<< "- "<<j<<_SVD(m_svd) << endl;

      //((Scalar)g.at<double>(i,j))[0] = _SVD(m_svd);
      //cout << ">>>>>: "<<i <<"-"<<j <<": "<<_SVD(m_svd) << endl;
      r_lsbp.at<double>(i,j) = _SVD(m_svd);
      /*cout << "lsbp: "<< r_lsbp.at<double>(i,j) << endl;
      cout << "SVD: "<<_SVD(m_svd) << endl;*/
      //cout << m_svd << endl;

      //waitKey(10000);

      //cout << ">>>>>: "<<i <<"-"<<j <<": "<<((Scalar)g.at<double>(i,j))[0] << endl;
      //waitKey(100);
    }
    
  }
  auto t12 = std::chrono::high_resolution_clock::now();
  cout << "Time_ex: " << std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t11).count() << endl;

  intensity_fr.release();
  
  /*for(int i=1; i<width-1; i++)
  {
    for(int j=1; j<heigth-1; j++)
    {
      if(abs(g.at<double>(i,j)-g.at<double>(i-1,j-1))<tau)
      {
        ((Scalar)LSBP.at<uchar>(i+1,j+1))[0] = 0;
      }
      else
      {
        ((Scalar)LSBP.at<uchar>(i+1,j+1))[0] = 1;
      }

    }
  }*/
}

Mat SVD_step(Mat frame, int threshold=4, int matches=2, int Rscale=5, double Rlr=0.1, double Tlr=0.02)
{
  Mat svd = Mat::zeros(frame.rows+2, frame.cols+2, CV_32FC1);
  extract_LSBP(frame, svd, 0.05);

  Mat white = Mat::ones(1,1, CV_8UC1)*255;
  Mat mask = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
  for(int i=0; i<frame.rows; i++)
  {
    for(int j=0; j<frame.cols; j++)
    {
      list<Mat>::iterator next_frame;
      next_frame = samples_frame.begin();

      list<Mat>::iterator next_lsbp;
      next_lsbp = samples_lsbp.begin();

      int samples_matches = 0;

      while(next_frame != samples_frame.end())
      {
        double L1_distance = abs(((Scalar)frame.at<Vec3b>(i, j))[0]-((Scalar)(*next_frame).at<Vec3b>(i, j))[0])+
        abs(((Scalar)frame.at<Vec3b>(i, j))[1]-((Scalar)(*next_frame).at<Vec3b>(i, j))[1])+
        abs(((Scalar)frame.at<Vec3b>(i, j))[2]-((Scalar)(*next_frame).at<Vec3b>(i, j))[2]);

        int d_hamming = Hamming_distance(svd, *next_lsbp, i+1, j+1, 0.05);
      

        //if((L1_distance < ((Scalar)R.at<float>(i, j))[0]) && (d_hamming < threshold))             
        //if((d_hamming < threshold))
        if((L1_distance < R.at<double>(i, j)))            
        {
          samples_matches++;
        }
        //imshow("imagen", *next_frame);
        next_frame++;
        next_lsbp++;
        //waitKey(5000);
      }


      if(samples_matches < matches)
      {
        mask.at<uchar>(i, j) = white.at<uchar>(0, 0);
      }
    }
  } 

  return mask;
}

int Hamming_distance(Mat svd_frame, Mat svd_sample, int i, int j, double tau)
{
  int hamming = 0;
  //if((abs((svd_frame.at<double>(i,j))-(svd_frame.at<double>(i-1,j-1))) < tau))
  if((abs((svd_frame.at<double>(i,j))-(svd_frame.at<double>(i-1,j-1))) < tau) != (abs((svd_sample.at<double>(i,j))-(svd_sample.at<double>(i-1,j-1))) < tau))
  {
    hamming++;
  }
  if((abs(svd_frame.at<double>(i,j)-svd_frame.at<double>(i-1,j)) < tau) != (abs(svd_sample.at<double>(i,j)-svd_sample.at<double>(i-1,j)) < tau))
  {
    hamming++;
  }
  if((abs(svd_frame.at<double>(i,j)-svd_frame.at<double>(i-1,j+1)) < tau) != (abs(svd_sample.at<double>(i,j)-svd_sample.at<double>(i-1,j+1)) < tau))
  {
    hamming++;
  }
  if((abs(svd_frame.at<double>(i,j)-svd_frame.at<double>(i,j-1)) < tau) != (abs(svd_sample.at<double>(i,j)-svd_sample.at<double>(i,j-1)) < tau))
  {
    hamming++;
  }
  if((abs(svd_frame.at<double>(i,j)-svd_frame.at<double>(i,j+1)) < tau) != (abs(svd_sample.at<double>(i,j)-svd_sample.at<double>(i,j+1)) < tau))
  {
    hamming++;
  }
  if((abs(svd_frame.at<double>(i,j)-svd_frame.at<double>(i+1,j-1)) < tau) != (abs(svd_sample.at<double>(i,j)-svd_sample.at<double>(i+1,j-1)) < tau))
  {
    hamming++;
  }
  if((abs(svd_frame.at<double>(i,j)-svd_frame.at<double>(i+1,j)) < tau) != (abs(svd_sample.at<double>(i,j)-svd_sample.at<double>(i+1,j)) < tau))
  {
    hamming++;
  }
  if((abs(svd_frame.at<double>(i,j)-svd_frame.at<double>(i+1,j+1)) < tau) != (abs(svd_sample.at<double>(i,j)-svd_sample.at<double>(i+1,j+1)) < tau))
  {
    hamming++;
  }

  //cout << i<<"-"<< j <<": "<<abs(((Scalar)svd_frame.at<double>(i,j))[0]-((Scalar)svd_frame.at<double>(i-1,j-1))[0]) << " - " << abs(((Scalar)svd_sample.at<double>(i,j))[0]-((Scalar)svd_sample.at<double>(i-1,j-1))[0]) << " ham: "<< hamming<<endl;
  //cout << i<<"-"<< j <<": "<< abs(((Scalar)svd_frame.at<double>(i,j))[0]-((Scalar)svd_frame.at<double>(i-1,j-1))[0]) <<" Bool: "<<(abs(((Scalar)svd_frame.at<double>(i,j))[0]-((Scalar)svd_frame.at<double>(i-1,j-1))[0])<0.05)  << " - " << " ham: "<< hamming<< " tau: " <<tau<<endl;
  //waitKey(10);

  return hamming;
}

double _SVD(arma::mat matriz)
{
  /*double a = 2;
  arma::mat m;
  m = {{0.0, 1.0, a},
    { a, 0.0, a },
    { a, -2.0, 1.0 }};*/
  
    arma::mat U2, V2;
    arma::vec w2;
    
    arma::svd(U2, w2, V2, matriz);

    //Mat opencv_mat(matriz.rows, matriz.cols, CV_64FC1, U2.memptr());

    //cout << w2[2] << " "<<w2[1] << " "<<w2[0] << endl;
  return ((w2[2]+w2[1])/w2[0]);
  //return singular_values; 
}

int clip(int i, int inferior, int superior, int val_range)
{
  int i0;
  if(i<inferior)
    {
      i0 = rand()%val_range-rand()%val_range+inferior;
    }
    else
    {
      if(i>superior)
      {
        i0 = rand()%val_range-rand()%val_range+superior;
      }
      else
      {
        i0 = rand()%val_range-rand()%val_range+i;
      }
    }

  return i0;

}