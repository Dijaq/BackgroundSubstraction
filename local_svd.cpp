#include <iostream>
#include <string.h>
#include <experimental/filesystem>
#include <list>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <armadillo>
#include <chrono>

using namespace std;
using namespace cv;

namespace fs = std::experimental::filesystem::v1;

void extract_LSBP(Mat frame, Mat &output, int tau);
Mat SVD_init(Mat frame, int samples);
Mat _SVD_init(Mat frame);
Mat SVD_step(Mat, int, int, int, double, double);
double _SVD(arma::mat matriz);// return the singular values sum (s[1]+s[2])/s[0]
int clip(int i, int inferior, int superior, int val_range);


Mat D, fr, lsbp;
list<Mat> samples_lsbp;
list<Mat> samples_frame;
int heigth, width;

int main()
{
	auto duration =0;

	int samples = 10;
	/*list<string> imagenes;
	string path = "highway/input/";
	for(auto & p : fs::directory_iterator(path))
		string a = p.string().c_str();
		imagenes.push_back(to_string(p));
	
	imagenes.sort();
	auto it;
	for(it=imagenes.begin(); it != imagenes.end(); ++it)
		cout << *it << endl;
	cout << "Hola c++" << endl;*/

	Mat img;
	img = imread("highway/input/in000001.jpg", CV_LOAD_IMAGE_COLOR);
	heigth = img.cols;
	width = img.rows;

	Mat ones = Mat::ones(2, 3, CV_32F)*0.2;
	Mat R = Mat::ones(width, heigth, CV_8UC1)*0.2;
	Mat T = Mat::ones(width, heigth, CV_8UC1)*0.08;

	list<Mat> lD;
	for(int s=0; s<samples; s++)
	{
		D = Mat::ones(2, 2, CV_32F)*0.2;
		lD.push_back(D);
	}

//Size of the list
//	cout << lD.size() << endl;

//Print all the elements of the list
	list<Mat>::iterator next;
	next = lD.begin();
	while(next != lD.end())
	{
		//cout << *next << endl;
		next++;
	}


	namedWindow("imagen", WINDOW_AUTOSIZE);
	imshow("imagen", img);
	Mat result = SVD_init(img, samples);
	waitKey(5000);

	for(int f=2; f<=2; f++)
	{
		cout << "=========: " << f << endl;
		//Only to read
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
		//Delete for a video
		auto t11 = std::chrono::high_resolution_clock::now();
		//Mat result = SVD_init(img);

		//Print the frames generated
		/*list<Mat>::iterator next_frame;
		next_frame = samples_frame.begin();

		while(next_frame != samples_frame.end())
		{
			cout << "frames" << endl;
			imshow("imagen", *next_frame);
			next_frame++;
			waitKey(5000);
		}*/

		SVD_step(img, 4, 2, 5, 0.1, 0.02);

		auto t12 = std::chrono::high_resolution_clock::now();
		//Mat result = SVD_step(img); 
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t11).count();
		cout << "Time: " << duration << endl;
		
		//imshow("imagen", result);
		//waitKey(5000);
	}
	return 0;
}

//Extrae la matriz de valores singulares SVD (s[1]+s[2])/s[0]
void extract_LSBP(Mat frame, Mat &g, int tau=0.05)
{
	Mat intensity;
	cvtColor(frame, intensity, COLOR_BGR2GRAY);
//Falta asignar los bordes

	//Mat m_svd = Mat::zeros(3, 3, CV_8UC1);
	
	//Mat LSBP = Mat::zeros(width, heigth, CV_8UC9);
	
//SVD descomposicion
	auto t11 = std::chrono::high_resolution_clock::now();
	for(int i=1; i<width-1; i++)
	{
		for(int j=1; j<heigth-1; j++)
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
			m_svd = {{((Scalar)intensity.at<uchar>(i-1,j-1))[0],
			((Scalar)intensity.at<uchar>(i-1,j))[0],
			((Scalar)intensity.at<uchar>(i-1,j+1))[0]},

			{((Scalar)intensity.at<uchar>(i,j-1))[0],
			((Scalar)intensity.at<uchar>(i,j))[0],
			((Scalar)intensity.at<uchar>(i,j+1))[0]},

			{((Scalar)intensity.at<uchar>(i+1,j-1))[0],
			((Scalar)intensity.at<uchar>(i+1,j))[0],
			((Scalar)intensity.at<uchar>(i+1,j+1))[0]}};

			((Scalar)g.at<double>(i,j))[0] = _SVD(m_svd);
			//cout << m_svd << endl;

			//waitKey(10000);

			//cout << i <<"-"<<j <<": "<<((Scalar)g.at<double>(i,j))[0] << endl;
			//waitKey(10000);
		}
		
	}
	auto t12 = std::chrono::high_resolution_clock::now();
	cout << "Time_ex: " << std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t11).count() << endl;

	intensity.release();

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

Mat SVD_init(Mat frame, int samples)
{
	Mat svd = Mat::zeros(width, heigth, CV_8UC1);
	
	extract_LSBP(frame, svd, 0.05);

		//Mat result = SVD_step(img); 

	samples_lsbp.push_back(svd);
	samples_frame.push_back(frame);
	int i0, j0;

	for(int k=1; k<samples;k++)
	{
		//cout << "--------1 " << svd.rows << " "<<svd.cols<<endl;
		svd.copyTo(lsbp);
		//lsbp = Mat::zeros(4, 4, CV_8UC3);
		//cout << "--------2 "  <<lsbp.rows<< endl;
		//fr = Mat::zeros(width, heigth, CV_8UC3);
		fr = frame.clone();
		//cout << "--------3" << endl;
		//Mat fr = Mat::zeros(svd.rows, svd.cols, CV_8UC3);
		for(int i=0; i<frame.rows; i++)
		{
			for(int j=0; j<frame.cols; j++)
			{
				i0 = clip(i,30,frame.rows-30,30);
				j0 = clip(j,30,frame.cols-30,30);
				fr.at<Vec3b>(i0,j0) = frame.at<Vec3b>(i0, j0);
				((Scalar)lsbp.at<double>(i0,j0))[0] = ((Scalar)svd.at<double>(i0, j0))[0]; 
			}
		}

		samples_lsbp.push_back(lsbp);
		samples_frame.push_back(fr);

		lsbp.release();
		fr.release();

	}
	//Mat lsbp = extract_LSBP(frame);

	//return extract_LSBP(frame, 0.05);

	return frame;
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

Mat SVD_step(Mat frame, int threshold=4, int matches=2, int Rscale=5, double Rlr=0.1, double Tlr=0.02)
{
	list<Mat>::iterator next_frame;
	next_frame = samples_frame.begin();

	while(next_frame != samples_frame.end())
	{
		cout << "frames" << endl;
		imshow("imagen", *next_frame);
		next_frame++;
		waitKey(5000);
	}

	//return extract_LSBP(frame, frame, 0.05);
	return frame;
}

//return calcular los valores singulares y retorna (s[2]+s[1])/s[0]
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