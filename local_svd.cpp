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
int Hamming_distance(Mat svd_frame, Mat svd_sample, int i, int j, double tau);


Mat D, fr, lsbp;
Mat R, T;
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

	Mat ones = Mat::ones(2, 3, CV_32FC1)*0.2;
	R = Mat::ones(width, heigth, CV_32FC1)*3.0;
	T = Mat::ones(width, heigth, CV_8UC1)*0.08;

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
	waitKey(1);

	/*list<Mat>::iterator next_f;
	next_f = samples_frame.begin();
	while(next_f != samples_frame.end())
	{
		cout << "------------------------" << endl;
		for(int j=0; j<(*next_f).cols; j++)
		{
			if(j<10)
			{
				cout << "...>" << (*next_f).at<Vec3b>(10,j)<<endl;
			}
		}
		imshow("imagen", *next_f);
		waitKey(1000);
		//cout << *next << endl;
		next_f++;
	}*/

	for(int f=2; f<=40; f++)
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

		Mat result = SVD_step(img, 4, 2, 5, 0.1, 0.02);
		imshow("imagen", result);
		waitKey(5);


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

Mat SVD_init(Mat frame, int samples)
{
	Mat svd = Mat::zeros(width+2, heigth+2, CV_32FC1);
	
	extract_LSBP(frame, svd, 0.05);

	//cout << "Impr1" << endl;

		//Mat result = SVD_step(img); 

	samples_lsbp.push_back(svd);
	//cout << "Impr2" << endl;
	samples_frame.push_back(frame);
	//cout << "Impr3" << endl;
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
				i0 = clip(i,10,frame.rows-10,10);
				j0 = clip(j,10,frame.cols-10,10);

				fr.at<Vec3b>(i0,j0) = frame.at<Vec3b>(i,j);
				//((Scalar)fr.at<Vec3b>(i0,j0))[0] = ((Scalar)frame.at<Vec3b>(i, j))[0];
				//((Scalar)fr.at<Vec3b>(i0,j0))[1] = ((Scalar)frame.at<Vec3b>(i, j))[1];
				//((Scalar)fr.at<Vec3b>(i0,j0))[2] = ((Scalar)frame.at<Vec3b>(i, j))[2];
				//((Scalar)lsbp.at<double>(i0,j0))[0] = ((Scalar)svd.at<double>(i, j))[0]; 
				lsbp.at<double>(i0,j0) = svd.at<double>(i, j); 
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

//threshold  HR PY
// matches   threshold PY
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
				//double L1_distance = ((Scalar)(*next_frame).at<Vec3b>(i, j))[0];

				/*double L1_distance = abs(((Scalar)frame.at<Vec3b>(i, j))[0]-((Scalar)(*next_frame).at<Vec3b>(i, j))[0])+
				abs(((Scalar)frame.at<Vec3b>(i, j))[1]-((Scalar)(*next_frame).at<Vec3b>(i, j))[1])+
				abs(((Scalar)frame.at<Vec3b>(i, j))[2]-((Scalar)(*next_frame).at<Vec3b>(i, j))[2]);*/
				double L1_distance = abs(((Scalar)frame.at<Vec3b>(i, j))[0]-((Scalar)(*next_frame).at<Vec3b>(i, j))[0])+
				abs(((Scalar)frame.at<Vec3b>(i, j))[1]-((Scalar)(*next_frame).at<Vec3b>(i, j))[1])+
				abs(((Scalar)frame.at<Vec3b>(i, j))[2]-((Scalar)(*next_frame).at<Vec3b>(i, j))[2]);

				int d_hamming = Hamming_distance(svd, *next_lsbp, i+1, j+1, 0.05);
				//*next_lsbp;

				//fr.at<Vec3b>(i,j) = frame.at<Vec3b>(i, j);

				/*if(i==10)
				{
					if(j<10)
					{
						cout << i <<" - "<< j <<" L1 : "<< L1_distance << " R "<<((*next_frame).at<Vec3b>(i, j))<<endl;
						waitKey(10);
					}
				}*/

				//if((L1_distance < ((Scalar)R.at<float>(i, j))[0]) && (d_hamming < threshold))							
				//if((d_hamming < threshold))
				if((L1_distance < R.at<double>(i, j)))						
				{
					samples_matches++;
				}
				//cout << i<<" - " << j << ": " << samples_matches<< " -- " << L1_distance<<endl;
				//waitKey(100);

				//cout << "frames" << endl;
				//imshow("imagen", *next_frame);
				next_frame++;
				next_lsbp++;
				//waitKey(5000);
			}

			//cout << "Samples match: " << i<<"-"<< j<<" : "<< ((Scalar)svd.at<uchar>(i,j))[0] << endl;
			//waitKey(10);

			if(samples_matches < matches)
			{
				//cout << "Samples match: " << i<<"-"<< j<<" : "<< ((Scalar)mask.at<uchar>(i, j))[0] <<" -- "<<samples_matches << endl;
				mask.at<uchar>(i, j) = white.at<uchar>(0, 0);
				//cout << "-->" << ((Scalar)mask.at<uchar>(i, j))[0] << " - "<< ((Scalar)white.at<uchar>(0, 0))[0] <<endl;
				//waitKey(10);
			}
		}
	}	

	//return extract_LSBP(frame, frame, 0.05);
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