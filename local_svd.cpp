#include <iostream>
#include <string.h>
#include <experimental/filesystem>
#include <list>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <armadillo>

using namespace std;
using namespace cv;

namespace fs = std::experimental::filesystem::v1;

Mat extract_LSBP(Mat frame);
Mat SVD_init(Mat frame);
double _SVD(arma::mat matriz);// return the singular values sum (s[1]+s[2])/s[0]

Mat D, samples_lsbp;
list<Mat> samples_int;
int heigth, width;

int main()
{

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
	Mat R = Mat::ones(width, heigth, CV_32F)*0.2;
	Mat T = Mat::ones(width, heigth, CV_32F)*0.08;

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

		Mat result = SVD_init(img); 
		
		imshow("imagen", result);

		waitKey(5000);
	}
	return 0;
}


Mat extract_LSBP(Mat frame, int tau=0.05)
{
	Mat intensity;
	cvtColor(frame, intensity, COLOR_BGR2GRAY);
//Falta asignar los bordes

	//Mat m_svd = Mat::zeros(3, 3, CV_8UC1);
	Mat g = Mat::zeros(width, heigth, CV_8UC1);
	
//SVD descomposicion
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

			//cout << m_svd << endl;

			//waitKey(10000);

			g.at<double>(i,j) = _SVD(m_svd);
			//cout << i <<"-"<<j <<": "<<((Scalar)g.at<double>(i,j))[0] << endl;
			//waitKey(10000);
		}
	}

	return g;
}

Mat SVD_init(Mat frame)
{
	//Mat lsbp = extract_LSBP(frame);

	return extract_LSBP(frame, 0.05);
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