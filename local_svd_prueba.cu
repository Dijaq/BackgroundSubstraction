#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include <string.h>
#include <list>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <armadillo>
#include <chrono>

//[offset*3+0] [offset*3+1] [offset*3+2]

using namespace std;
using namespace cv;

void extract_LSBP(Mat frame, Mat &output, int tau);
Mat SVD_init(Mat frame, int samples);
Mat _SVD_init(Mat frame);
Mat SVD_step(Mat, int, int, int, double, double);
double _SVD(arma::mat matriz);// return the singular values sum (s[1]+s[2])/s[0]
int clip(int i, int inferior, int superior, int val_range);
int Hamming_distance(Mat svd_frame, Mat svd_sample, int i, int j, double tau);

//Para CUDA
void SVD_init_matrices(Mat frame, int samples);
Mat SVD_step_matrices(Mat, int, int, int, double, double);
int ** Mat_to_matriz(Mat img);
Mat matriz_to_Mat(int **img, int rows, int cols);
int Mat_to_matriz_c(Mat img);

Mat D, fr, lsbp;
Mat R, T;
list<Mat> samples_lsbp;
list<Mat> samples_frame;
list<int**> samples_frame_cu;
int heigth, width;

#define filas_cu 240
#define columnas_cu 320
#define Threads 64 // max threads x bloque
#define Bloques 20

__global__ void sumaMatrices(int *m1, int *m2, int *m3) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int fil = blockIdx.y * blockDim.y + threadIdx.y;

  int indice = fil * columnas_cu + col;


  if (col < columnas_cu && fil < filas_cu) {
  // debido a que en los últimos bloques no se realizan todos los threads
  	m3[indice] = m1[indice] + m2[indice];
  }
}


int main()
{
	//auto duration =0;

	int samples = 10;

	Mat img;
	img = imread("highway/input/in000001.jpg", CV_LOAD_IMAGE_COLOR);
	heigth = img.cols;
	width = img.rows;

	Mat ones = Mat::ones(2, 3, CV_32FC1)*0.2;
	R = Mat::ones(width, heigth, CV_32FC1)*3.0;
	T = Mat::ones(width, heigth, CV_8UC1)*0.08;

	/*list<Mat> lD;
	for(int s=0; s<samples; s++)
	{
		D = Mat::ones(2, 2, CV_32F)*0.2;
		lD.push_back(D);
	}


	list<Mat>::iterator next;
	next = lD.begin();
	while(next != lD.end())
	{
		next++;
	}*/

	/*cvtColor(img, img, COLOR_BGR2GRAY);
	int **matriz = Mat_to_matriz(img);
	Mat imagen = matriz_to_Mat(matriz, img.rows, img.cols);*/

	SVD_init_matrices(img, samples);
	//namedWindow("imagen", WINDOW_AUTOSIZE);
	//imshow("imagen", imagen);
	//Mat result = SVD_init(img, samples);
	//waitKey(0);



	for(int f=2; f<=40; f++)
	{
		cout << "=========: " << f << endl;
		if(f<10)
			img = imread("highway/input/in00000"+to_string(f)+".jpg", CV_LOAD_IMAGE_COLOR);
		else
			if(f<100)
				img = imread("highway/input/in0000"+to_string(f)+".jpg", CV_LOAD_IMAGE_COLOR);
			else
				if(f<1000)
					img = imread("highway/input/in000"+to_string(f)+".jpg", CV_LOAD_IMAGE_COLOR);
				else
					img = imread("highway/input/in00"+to_string(f)+".jpg", CV_LOAD_IMAGE_COLOR);

		auto t11 = std::chrono::high_resolution_clock::now();

		Mat result = SVD_step_matrices(img, 5, 2, 5, 0.1, 0.02);
		imshow("imagen", result);
		waitKey(1);


		auto t12 = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t11).count();
		cout << "Time of proccess: " << duration << endl;
		
		
	}
	return 0;
}

int **Mat_to_matriz(Mat img)
{
	int **matriz = new int*[img.rows]; 
	for(int i=0; i<img.rows; i++)
	{
		matriz[i] = new int[img.cols]; 
		for(int j=0; j<img.cols; j++)
		{
			//cout << img.rows <<" - " << img.cols << endl;
			matriz[i][j] = img.at<uchar>(i, j);
		}
	}

	return matriz;
}


Mat matriz_to_Mat(int **img, int rows, int cols)
{
	Mat image = Mat::zeros(rows, cols, CV_8UC1);
	for(int i=0; i< rows; i++)
	{
		for(int j=0; j<cols; j++)
		{
			cout << i << "-" << j << endl;
			image.at<uchar>(i, j) = img[i][j];			
		}
	}

	return image;
}



//Extrae la matriz de valores singulares SVD (s[1]+s[2])/s[0]
//intensity_fr with 0 
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

	
//SVD descomposicion
	auto t11 = std::chrono::high_resolution_clock::now();
#pragma omp parallell for
	for(int i=1; i<intensity_fr.rows-1; i++)
	{
		for(int j=1; j<intensity_fr.cols-1; j++)
		{
			
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

			r_lsbp.at<double>(i,j) = _SVD(m_svd);
		}
		
	}
	auto t12 = std::chrono::high_resolution_clock::now();
	cout << "Time_ex: " << std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t11).count() << endl;

	intensity_fr.release();
	
}

Mat SVD_init(Mat frame, int samples)
{
	Mat svd = Mat::zeros(width+2, heigth+2, CV_32FC1);
	
	extract_LSBP(frame, svd, 0.05);

	samples_lsbp.push_back(svd);
	//cout << "Impr2" << endl;
	samples_frame.push_back(frame);

	//TO CUDA
	Mat gray;
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	int **matriz_gray = Mat_to_matriz(gray);
	samples_frame_cu.push_back(matriz_gray);
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
			
			}
		}

		extract_LSBP(fr, lsbp, 0.05);

		samples_lsbp.push_back(lsbp);
		samples_frame.push_back(fr);
		//CUDA
		Mat gray;
		cvtColor(fr, gray, COLOR_BGR2GRAY);
		int **matriz_gray = Mat_to_matriz(gray);
		samples_frame_cu.push_back(matriz_gray);

		lsbp.release();
		fr.release();

	}

	return frame;
}

void SVD_init_matrices(Mat frame, int samples)
{
	Mat svd = Mat::zeros(width+2, heigth+2, CV_32FC1);
	
	extract_LSBP(frame, svd, 0.05);

	samples_lsbp.push_back(svd);
	//cout << "Impr2" << endl;
	samples_frame.push_back(frame);

	//TO CUDA
	Mat gray;
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	int **matriz_gray = Mat_to_matriz(gray);
	samples_frame_cu.push_back(matriz_gray);
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
			
			}
		}

		extract_LSBP(fr, lsbp, 0.05);

		samples_lsbp.push_back(lsbp);
		samples_frame.push_back(fr);
		//CUDA
		Mat gray;
		cvtColor(fr, gray, COLOR_BGR2GRAY);
		int **matriz_gray = Mat_to_matriz(gray);
		samples_frame_cu.push_back(matriz_gray);
		//CUDA

		lsbp.release();
		fr.release();

	}
}

Mat _SVD_init(Mat frame)
{
	Mat output = Mat::zeros(width, heigth, CV_8UC1);
	extract_LSBP(frame, output, 0.05);
	cout << "it1" << endl;
	Mat _frame = frame.clone();
	cout << "it2" << endl;
	return _frame;
}

//threshold  HR PY
// matches   threshold PY
Mat SVD_step(Mat frame, int threshold=4, int matches=2, int Rscale=5, double Rlr=0.1, double Tlr=0.02)
{
	Mat svd_fr = Mat::zeros(width+2, heigth+2, CV_32FC1);
	extract_LSBP(frame, svd_fr, 0.05);

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

			while(next_lsbp != samples_lsbp.end())
			{			
				double L1_distance = abs(((Scalar)frame.at<Vec3b>(i, j))[0]-((Scalar)(*next_frame).at<Vec3b>(i, j))[0])+
				abs(((Scalar)frame.at<Vec3b>(i, j))[1]-((Scalar)(*next_frame).at<Vec3b>(i, j))[1])+
				abs(((Scalar)frame.at<Vec3b>(i, j))[2]-((Scalar)(*next_frame).at<Vec3b>(i, j))[2]);

				int d_hamming = Hamming_distance(svd_fr, *next_lsbp, i+1, j+1, 0.05);
				
				
				//if((L1_distance < ((Scalar)R.at<float>(i, j))[0]) && (d_hamming < threshold))							
				//if((d_hamming < threshold))
				if((L1_distance < R.at<double>(i, j)))						
				{
					samples_matches++;
				}

				next_frame++;
				next_lsbp++;
			}

			if(samples_matches < matches)
			{
				mask.at<uchar>(i, j) = 255;
			}
		}
		
	}	

	return mask;
}

Mat SVD_step_matrices(Mat frame, int threshold=4, int matches=2, int Rscale=5, double Rlr=0.1, double Tlr=0.02)
{
	Mat gray;
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	int **m1 = Mat_to_matriz(gray);
	int **m2 = Mat_to_matriz(gray);
	int **m3 = Mat_to_matriz(gray);

	int filas = frame.rows;
	int columnas = frame.cols;
	int *dm1, *dm2, *dm3;

	cudaMalloc((void**) &dm1, filas * columnas * sizeof(int));
	cudaMalloc((void**) &dm2, filas * columnas * sizeof(int));
	cudaMalloc((void**) &dm3, filas * columnas * sizeof(int));

	// copiando memoria a la GPGPU
	cudaMemcpy(dm1, m1, filas * columnas * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dm2, m2, filas * columnas * sizeof(int), cudaMemcpyHostToDevice);

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
	cudaMemcpy(m3, dm3, filas * columnas * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(m2, dm2, N * N * sizeof(int), cudaMemcpyDeviceToHost);

	Mat mask = matriz_to_Mat(m3, filas, columnas);
	cout << "SUMA MATRICES" << endl;

	cudaFree(dm1);
	cudaFree(dm2);
	cudaFree(dm3);

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

	return hamming;
}

//return calcular los valores singulares y retorna (s[2]+s[1])/s[0]
double _SVD(arma::mat matriz)
{
	
    arma::mat U2, V2;
    arma::vec w2;
    
    arma::svd(U2, w2, V2, matriz);

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
