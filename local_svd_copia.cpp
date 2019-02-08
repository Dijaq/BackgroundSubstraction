#include <iostream>
#include <string.h>
#include <experimental/filesystem>
#include <list>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <armadillo>
#include <chrono>
#include <time.h>

using namespace std;
using namespace cv;

namespace fs = std::experimental::filesystem::v1;

void extract_LSBP(Mat frame, Mat &output, int tau);
Mat SVD_init(Mat frame, int samples);
Mat _SVD_init(Mat frame, int samples);
Mat SVD_step(Mat, int, int, int, double, double);
double _SVD(arma::mat matriz);// return the singular values sum (s[1]+s[2])/s[0]
int clip(int i, int inferior, int superior, int val_range);
int Hamming_distance(Mat svd_frame, Mat svd_sample, int i, int j, double tau);
void export_mat_excel(Mat img);
void update_samples_lsbp();


Mat D, fr, lsbp;
Mat R, T;
list<Mat> samples_lsbp;
list<Mat> samples_frame;
int heigth, width;

int main()
{
	srand(time(NULL));
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
	R = Mat::ones(width, heigth, CV_32FC1)*80.0;
	
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

	for(int f=2; f<=100; f++)
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

		Mat result = SVD_step(img, 5, 2, 5, 0.1, 0.02);
		/*SVD_step(img, 5, 2, 5, 0.1, 0.02);

		list<Mat>::iterator result_frame;
		result_frame = samples_frame.begin();
		result_frame++;
		result_frame++;
		result_frame++;*/
		imshow("imagen", result);
		waitKey(5);


		auto t12 = std::chrono::high_resolution_clock::now();
		//Mat result = SVD_step(img); 
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t11).count();
		cout << "Time of proccess: " << duration << endl;
		
		//imshow("imagen", result);
		//waitKey(5000);
	}
	return 0;
}


void export_mat_excel(Mat img)
{
	ofstream myfile;
  	myfile.open("example.csv");
	for(int i=0; i<img.rows; i++)
	{
		for(int j=0; j<img.cols; j++)
		{
			if(i==2 && j==1)
			{
				cout << "print: " << ((Scalar)img.at<float>(i,j))[0] << endl;
			}
			myfile << ((Scalar)img.at<float>(i, j))[0];
			myfile << ",";
		}
		myfile << "\n";

	}
	myfile.close();
	waitKey(5000);
}


//Extrae la matriz de valores singulares SVD (s[1]+s[2])/s[0]
//intensity_fr with 0 
void extract_LSBP(Mat frame, Mat &r_lsbp, int tau=0.05)
{
	//Mat other = Mat::zeros(width+2, heigth+2, CV_32FC1);
	Mat intensity;
	cvtColor(frame, intensity, COLOR_BGR2GRAY);
	Mat intensity_fr = Mat::zeros(frame.rows+2, frame.cols+2, CV_8UC1);

//#pragma omp parallell for
	for(int i=1; i<intensity_fr.rows-1; i++)
	{
		for(int j=1; j<intensity_fr.cols-1; j++)
		{
			intensity_fr.at<uchar>(i,j) = intensity.at<uchar>(i-1,j-1); 
		}
	}

	//imshow("imagen", intensity_fr);
	//waitKey(5000);


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

			r_lsbp.at<float>(i,j) = _SVD(m_svd);

			/*if(j == 1)
			{
				ofstream myfile;
			  	myfile.open("example.csv");
				for(int i=0; i<r_lsbp.rows; i++)
				{
					for(int j=0; j<r_lsbp.cols; j++)
					{
						if(i==2 && j==1)
							cout << "print: " << ((Scalar)r_lsbp.at<double>(i,j))[0] << endl;
						myfile << ((Scalar)r_lsbp.at<double>(i, j))[0];
						myfile << ",";
					}
					myfile << "\n";
				}
				
				imshow("imagen", intensity_fr);
					//cout << "-> " << ((Scalar)mask.at<uchar>(15, j))[0] << endl;
				myfile.close();
				waitKey(5000);
			}*/
			/*if(i==2 && j==1)
			{
				cout << "SVD: "<<_SVD(m_svd) << endl;
				cout << "lsbp: " << r_lsbp.at<double>(i,j) << endl;
				cout << i <<" - "<<j<< " : "<<((Scalar)intensity_fr.at<uchar>(i-1,j-1))[0]<<endl;
				cout << i <<" - "<<j<< " : "<<((Scalar)intensity_fr.at<uchar>(i-1,j))[0]<<endl;
				cout << i <<" - "<<j<< " : "<<((Scalar)intensity_fr.at<uchar>(i-1,j+1))[0]<<endl;
				cout << i <<" - "<<j<< " : "<<((Scalar)intensity_fr.at<uchar>(i,j-1))[0]<<endl;
				cout << i <<" - "<<j<< " : "<<((Scalar)intensity_fr.at<uchar>(i,j))[0]<<endl;
				cout << i <<" - "<<j<< " : "<<((Scalar)intensity_fr.at<uchar>(i,j+1))[0]<<endl;
				cout << i <<" - "<<j<< " : "<<((Scalar)intensity_fr.at<uchar>(i+1,j-1))[0]<<endl;
				cout << i <<" - "<<j<< " : "<<((Scalar)intensity_fr.at<uchar>(i+1,j))[0]<<endl;
				cout << i <<" - "<<j<< " : "<<((Scalar)intensity_fr.at<uchar>(i+1,j+1))[0]<<endl;
				waitKey(5000);
			}*/
			//cout << ">>>>>: "<<i <<"-"<<j <<": "<<((Scalar)g.at<double>(i,j))[0] << endl;
			
		}

		
		
	}
	auto t12 = std::chrono::high_resolution_clock::now();
	//cout << "Time_ex: " << std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t11).count() << endl;

	intensity_fr.release();
	
}

Mat SVD_init(Mat frame, int samples)
{
	Mat svd = Mat::zeros(width+2, heigth+2, CV_32FC1);
	
	extract_LSBP(frame, svd, 0.05);

	/*ofstream myfile;
  	myfile.open("example.csv");
	for(int i=0; i<svd.rows; i++)
	{
		for(int j=0; j<svd.cols; j++)
		{
			if(i==2 && j==1)
				cout << "print: " << ((Scalar)svd.at<double>(i,j))[0] << endl;
			myfile << ((Scalar)svd.at<double>(i, j))[0];
			myfile << ",";
		}
		myfile << "\n";
	}
	
	imshow("imagen", svd);
		//cout << "-> " << ((Scalar)mask.at<uchar>(15, j))[0] << endl;
	myfile.close();
	waitKey(5000);*/

	samples_lsbp.push_back(svd);
	//cout << "Impr2" << endl;
	samples_frame.push_back(frame);
	//cout << "Impr3" << endl;
	int i0, j0;

#pragma omp parallell for
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

		lsbp.release();
		fr.release();

	}

	return frame;
}

Mat _SVD_init(Mat frame, int samples)
{
	Mat svd = Mat::zeros(width+2, heigth+2, CV_32F);

	//extract_LSBP(frame, svd, 0.05);
	svd.at<double>(1,1) = 1.24;

	ofstream myfile;
  	myfile.open("example.csv");
	for(int i=0; i<svd.rows; i++)
	{
		for(int j=0; j<svd.cols; j++)
		{
			if(i==2 && j==1)
				cout << "print: " << ((Scalar)svd.at<double>(i,j))[0] << endl;
			myfile << ((Scalar)svd.at<double>(i, j))[0];
			myfile << ",";
		}
		myfile << "\n";
	}
	
	imshow("imagen", svd);
		//cout << "-> " << ((Scalar)mask.at<uchar>(15, j))[0] << endl;
	myfile.close();
	waitKey(5000);

}

//threshold  HR PY
// matches   threshold PY
Mat SVD_step(Mat frame, int threshold=4, int matches=2, int Rscale=5, double Rlr=0.05, double Tlr=0.02)
{
	//Mat svd_fr = Mat::zeros(frame.rows+2, frame.cols+2, CV_32FC1);
	Mat svd_fr = Mat::zeros(width+2, heigth+2, CV_32FC1);
	extract_LSBP(frame, svd_fr, 0.05);

	Mat mask = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
	
	//Mat white = Mat::ones(1,1, CV_8UC1)*255;
//#pragma omp parallell for
	for(int i=0; i<frame.rows; i++)
	{
		for(int j=0; j<frame.cols; j++)
		{
			list<Mat>::iterator next_frame;
			next_frame = samples_frame.begin();

			list<Mat>::iterator next_lsbp;
			next_lsbp = samples_lsbp.begin();

			int samples_matches = 0;
			double L1_distance_sum = 0;
			//double L1_distance_min = 1000000;

			while(next_lsbp != samples_lsbp.end())
			{
				double L1_distance = abs(((Scalar)frame.at<Vec3b>(i, j))[0]-((Scalar)(*next_frame).at<Vec3b>(i, j))[0])+
				abs(((Scalar)frame.at<Vec3b>(i, j))[1]-((Scalar)(*next_frame).at<Vec3b>(i, j))[1])+
				abs(((Scalar)frame.at<Vec3b>(i, j))[2]-((Scalar)(*next_frame).at<Vec3b>(i, j))[2]);

				int d_hamming = Hamming_distance(svd_fr, *next_lsbp, i+1, j+1, 0.05);
		
				L1_distance_sum += L1_distance;
				//if(!((L1_distance < R.at<double>(i, j)) && (d_hamming < (threshold-1))))	

				//UPDATE R

				//if((L1_distance < R.at<float>(i, j)))						
				
				/*if(L1_distance_min > L1_distance)
				{
					L1_distance_min = L1_distance;
				}*/

				if((L1_distance < R.at<float>(i, j)) && (d_hamming < (threshold)))						
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

			//UPDATE R(x)
			/*if(R.at<float>(i, j) > (L1_distance_sum/10)*Rscale)
			{
				R.at<float>(i, j) = R.at<float>(i, j)*(1-Rlr);
			}
			else
			{
				R.at<float>(i, j) = R.at<float>(i, j)*(1+Rlr);
			}*/

			if(samples_matches < matches)
			{
				mask.at<uchar>(i, j) = 255;
			}
			else
			{
				//UPDATE B(X) because mask(i,j) is 0
				int random = rand()%10;
				
				list<Mat>::iterator next_frame_update;
				next_frame_update = samples_frame.begin();

				list<Mat>::iterator next_lsbp_update;
				next_lsbp_update = samples_lsbp.begin();

				for(int i=0; i<random; i++)
				{
					next_frame_update++;
					next_lsbp_update++;
				}
				(*next_frame_update).at<Vec3b>(i, j) = frame.at<Vec3b>(i,j);
				//(*next_lsbp_update).at<float>(i, j) = frame.at<float>(i,j);
			}
		}
		
	}	

	update_samples_lsbp();

	return mask;

}

void update_samples_lsbp()
{
	list<Mat>::iterator next_frame;
	next_frame = samples_frame.begin();

	list<Mat>::iterator next_lsbp;
	next_lsbp = samples_lsbp.begin();

	int samples_matches = 0;

	while(next_lsbp != samples_lsbp.end())
	{
		Mat svd = Mat::zeros(width+2, heigth+2, CV_32FC1);
		extract_LSBP(*next_frame, svd, 0.05);
		*next_lsbp = svd.clone();

		next_frame++;
		next_lsbp++;
	}
}

int Hamming_distance(Mat svd_frame, Mat svd_sample, int i, int j, double tau)
{
	int hamming = 0;
	//if((abs((svd_frame.at<double>(i,j))-(svd_frame.at<double>(i-1,j-1))) < tau))
	if((abs((svd_frame.at<float>(i,j))-(svd_frame.at<float>(i-1,j-1))) < tau) != (abs((svd_sample.at<float>(i,j))-(svd_sample.at<float>(i-1,j-1))) < tau))
	{
		hamming++;
	}
	if((abs(svd_frame.at<float>(i,j)-svd_frame.at<float>(i-1,j)) < tau) != (abs(svd_sample.at<float>(i,j)-svd_sample.at<float>(i-1,j)) < tau))
	{
		hamming++;
	}
	if((abs(svd_frame.at<float>(i,j)-svd_frame.at<float>(i-1,j+1)) < tau) != (abs(svd_sample.at<float>(i,j)-svd_sample.at<float>(i-1,j+1)) < tau))
	{
		hamming++;
	}
	if((abs(svd_frame.at<float>(i,j)-svd_frame.at<float>(i,j-1)) < tau) != (abs(svd_sample.at<float>(i,j)-svd_sample.at<float>(i,j-1)) < tau))
	{
		hamming++;
	}
	if((abs(svd_frame.at<float>(i,j)-svd_frame.at<float>(i,j+1)) < tau) != (abs(svd_sample.at<float>(i,j)-svd_sample.at<float>(i,j+1)) < tau))
	{
		hamming++;
	}
	if((abs(svd_frame.at<float>(i,j)-svd_frame.at<float>(i+1,j-1)) < tau) != (abs(svd_sample.at<float>(i,j)-svd_sample.at<float>(i+1,j-1)) < tau))
	{
		hamming++;
	}
	if((abs(svd_frame.at<float>(i,j)-svd_frame.at<float>(i+1,j)) < tau) != (abs(svd_sample.at<float>(i,j)-svd_sample.at<float>(i+1,j)) < tau))
	{
		hamming++;
	}
	if((abs(svd_frame.at<float>(i,j)-svd_frame.at<float>(i+1,j+1)) < tau) != (abs(svd_sample.at<float>(i,j)-svd_sample.at<float>(i+1,j+1)) < tau))
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