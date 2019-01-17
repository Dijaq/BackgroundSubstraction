#include <iostream>
#include <armadillo>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

double _SVD(Mat matriz);
void original();

int main()
{
	Mat img;
	img = imread("highway/input/in000001.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	Mat X = Mat::ones(3,3, CV_8UC1)*2.0;
	//cout << X << endl;
	int rows = X.rows;
	int cols = X.cols;
	//original();
	//Mat view = Mat::zeros(1,3,CV_64FC1);
	double view = _SVD(X);
	cout << "->" <<view << endl;
	
}

double _SVD(Mat matriz)
{
	double a = 2;
	arma::mat m;
	m = {{0.0, 1.0, a},
    { a, 0.0, a },
    { a, -2.0, 1.0 }};
	//cout << matriz << endl;
	//arma::mat arma_mat( reinterpret_cast<double*>(matriz.data), matriz.rows, matriz.cols);
	//Mat m = to_arma(matriz);
	//cout << arma_mat << endl;

    // Convert cv::Mat to arma::fmat
    //arma::fmat arma_img(reinterpret_cast<float*>(X.data), 3, 3);

    // ------ Perform SVD with Armadillo (0.05s)
    arma::mat U2, V2;
    arma::vec w2;
    arma::svd(U2, w2, V2, m);

    //Mat opencv_mat(matriz.rows, matriz.cols, CV_64FC1, U2.memptr());
    Mat singular_values(1, matriz.cols, CV_32F, w2.memptr());
    cout << w2[2] << endl;
    cout << singular_values << endl;
    cout << "<<<"<<(singular_values.at<double>(0,1)+singular_values.at<double>(0,2))/singular_values.at<double>(0,0) << endl;

	return ((w2[2]+w2[1])/w2[0]);
	//return singular_values; 
}

void original()
{
	Mat matriz = Mat::ones(3,3, CV_32F)*4.0;

	arma::mat arma_mat( reinterpret_cast<double*>(matriz.data), matriz.rows, matriz.cols);
	
    // Convert cv::Mat to arma::fmat
    //arma::fmat arma_img(reinterpret_cast<float*>(X.data), 3, 3);
    arma::mat A;
    A = {{1.0, 2.0, 3.0},
    { 2.0, 3.0, 5.0 },
    { 1.0, 3.0, 8.0 }};

   // arma::mat A = armaConv(X.data, X.rows, X.cols,false);
	//cout << arma_img.rows << endl;

    // Check if the image back from armadillo is okay
//    Mat opencv_img(arma_img.n_cols, arma_img.n_rows, CV_32FC1, arma_img.memptr());

    // ------ Perform SVD with OpenCV (2.5s)
/*    SVD svvd;
    Mat w1, U1, V1t;
    svvd.compute(opencv_img, w1, U1, V1t);

    Mat W1 = Mat::zeros(w1.rows, w1.rows, CV_32FC1);
    for (int i = 0; i<w1.rows; i++)
    {
        W1.at<float>(i, i) = w1.at<float>(i);
    }
    Mat opencv_img_result = U1 * W1 * V1t;
*/
    // ------ Perform SVD with Armadillo (0.05s)
    arma::mat U2, V2;
    arma::vec w2;
    arma::svd(U2, w2, V2, arma_mat);

    //Mat opencv_mat(matriz.rows, matriz.cols, CV_64FC1, U2.memptr());
    Mat opencv_w2(1, matriz.cols, CV_64FC1, w2.memptr());
    cout << opencv_w2<< endl;

    /*arma::fmat W2 = arma::zeros<arma::fmat>(rows, cols);
    for (int i = 0; i < cols; i++)
    {
        *(W2.memptr() + i * (1 + rows)) = *(w2.memptr() + i);
    }
    arma::fmat arma_img_result = U2 * W2* V2.t();*/
}
