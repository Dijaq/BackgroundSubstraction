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
#include <omp.h>
#include "svd3_cuda.h"
#include "stdio.h"

using namespace std;
using namespace cv;

namespace fs = std::experimental::filesystem::v1;

#define THREAD_MAX 4
#define Threads 32

void extract_LSBP(Mat frame, Mat &output, int tau);
void video(string);
void cdnet(string, int);
void extract_LSBP_v2(Mat frame, Mat change_frame, Mat last_lsbp, Mat &output, int tau);
Mat SVD_init(Mat frame, int samples);
Mat SVD_step(Mat, int, int, int, double, double);
double _SVD(arma::mat matriz);// return the singular values sum (s[1]+s[2])/s[0]
int clip(int i, int inferior, int superior, int val_range);
int Hamming_distance(Mat svd_frame, Mat svd_sample, int i, int j, double tau);
void export_mat_excel(Mat img, string name);
void update_samples_lsbp();
double get_distance_L1(double b1, double b2, double g1, double g2, double r1, double r2);
double min_distance_L1(double b1, double b2, double g1, double g2, double r1, double r2);
void init_change_lsbp();
void init_zeros_change_lsbp();
bool validate_change(float, float, float, float, float, float, float, float, float);
void* LSBP_parallel(void* arg);
void* LSBP1_parallel(void* arg);
void* SVD_step_parallell(void * arg);
void extract_LSBP_cuda(Mat frame, Mat &output);


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
    //m3[indexij] = s11+s22+s33;
    m3[indexij] = (s22+s33)/s11;
  }
}


Mat global_intensity_fr, global_change_frame, global_last_lsbp, global_output;
Mat global_frame, global_mask, global_svd;
int part=0;
Mat D, fr, lsbp;
Mat R, T;
list<Mat> samples_lsbp;
list<Mat> samples_frame;
list<Mat> samples_change_lsbp;
int heigth, width;

int main()
{
  string pathVideoName = "sanPablo/SanPabloVideo3.mp4";
  string pathCDnet = "office/";
  bool isVideo = false;
  if(isVideo)
    video(pathVideoName);
  else
  {
    int startFrame = 570;
    cdnet(pathCDnet,startFrame);
  }

  return 0;
}

void video(string pathVideo)
{
  string PATH = "peopleInShade/";
  srand(time(NULL));
  auto duration =0;

  int samples = 10;

  Mat img;
  //img = imread(PATH+"input/in000001.jpg", CV_LOAD_IMAGE_COLOR);
  
  VideoCapture cap(pathVideo);
  int i=0;//  Mat ones = Mat::ones(2, 3, CV_32FC1)*0.2;
  

  namedWindow("imagen", WINDOW_AUTOSIZE);
  namedWindow("input", WINDOW_AUTOSIZE);
  //imshow("imagen", img);
  //Mat result = SVD_init(img, samples);
  waitKey(1);

  init_change_lsbp();

  if(!cap.isOpened())
  {
    cout << "Error opening video stream" << endl;
  } 

  while(1)
  {
    Mat frame;
    cap >> frame;
    if(i==0)
    {

      heigth = frame.cols;
      width = frame.rows;
      R = Mat::ones(width, heigth, CV_32FC1)*30.0;
      D = Mat::ones(width, heigth, CV_32FC1)*0.0;
      T = Mat::ones(width, heigth, CV_32FC1)*0.08;
      cout << "heigth: " << R.cols << endl;
      cout << "width: " << width << endl;
      Mat result = SVD_init(frame, samples);
    }
    else
    {     
      //cout << "=========: " << i << endl;
      //Only to read
      auto t11 = std::chrono::high_resolution_clock::now();
      //waitKey(5000);
      //cout << "Step0" << endl;
      Mat result = SVD_step(frame, 6, 2, 5, 0.05, 0.02);
      imshow("imagen", result);
      imshow("input", frame);
      //cout << "Step1" << endl;
    
      
      char c=(char)waitKey(1);
      if(c==1)
        break;


      auto t12 = std::chrono::high_resolution_clock::now();
      
      duration = std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t11).count();
      cout << "Frame: " << i <<" Time: " << duration << endl;
    }
    i++;
  
  }
}

void cdnet(string PATH, int starFrame)
{
  //string PATH = "highway/";
  srand(time(NULL));
  auto duration =0;

  int samples = 10;

  Mat img;
  img = imread(PATH+"input/in000001.jpg", CV_LOAD_IMAGE_COLOR);
  heigth = img.cols;
  width = img.rows;

  Mat ones = Mat::ones(2, 3, CV_32FC1)*0.2;
  R = Mat::ones(width, heigth, CV_32FC1)*30.0;
  D = Mat::ones(width, heigth, CV_32FC1)*0.0;
  
  T = Mat::ones(width, heigth, CV_32FC1)*0.08;

  namedWindow("imagen", WINDOW_AUTOSIZE);
  namedWindow("input", WINDOW_AUTOSIZE);
  imshow("imagen", img);
  Mat result = SVD_init(img, samples);
  waitKey(1);

  init_change_lsbp();

  for(int f=starFrame; f<=1699; f++)
  {
    //Only to read
    if(f<10)
      img = imread(PATH+"input/in00000"+to_string(f)+".jpg", CV_LOAD_IMAGE_COLOR);
    else
      if(f<100)
        img = imread(PATH+"input/in0000"+to_string(f)+".jpg", CV_LOAD_IMAGE_COLOR);
      else
        if(f<1000)
          img = imread(PATH+"input/in000"+to_string(f)+".jpg", CV_LOAD_IMAGE_COLOR);
        else
          img = imread(PATH+"input/in00"+to_string(f)+".jpg", CV_LOAD_IMAGE_COLOR);
    
    auto t11 = std::chrono::high_resolution_clock::now();
    
    Mat result = SVD_step(img, 6, 2, 5, 0.05, 0.02);
  
    imshow("imagen", result);
    imshow("input", img);
    waitKey(1);


    auto t12 = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t11).count();
    cout << "Frame: " << f <<" Time: " << duration << endl;
    
  
  }
}


void export_mat_excel(Mat img, string name)
{
  ofstream myfile;
    myfile.open(name+".csv");
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
  //waitKey(5000);
}


//Extrae la matriz de valores singulares SVD (s[1]+s[2])/s[0]
//intensity_fr with 0 
/*void extract_LSBP(Mat frame, Mat &r_lsbp, int tau=0.05)
{
  Mat intensity;
  cvtColor(frame, intensity, COLOR_BGR2GRAY);
  Mat intensity_fr = Mat::zeros(frame.rows+2, frame.cols+2, CV_8UC1);

  for(int i=1; i<intensity_fr.rows-1; i++)
  {
    for(int j=1; j<intensity_fr.cols-1; j++)
    {
      intensity_fr.at<uchar>(i,j) = intensity.at<uchar>(i-1,j-1); 
    }
  }

  auto t11 = std::chrono::high_resolution_clock::now();

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
    }
    
  }
  auto t12 = std::chrono::high_resolution_clock::now();

  intensity_fr.release();
}*/

void extract_LSBP(Mat frame, Mat &r_lsbp, int tau=0.05)
{
  Mat intensity;
  cvtColor(frame, intensity, COLOR_BGR2GRAY);
  Mat intensity_fr = Mat::zeros(frame.rows+2, frame.cols+2, CV_8UC1);

  for(int i=1; i<intensity_fr.rows-1; i++)
  {
    for(int j=1; j<intensity_fr.cols-1; j++)
    {
      intensity_fr.at<uchar>(i,j) = intensity.at<uchar>(i-1,j-1); 
    }
  }

  global_intensity_fr = intensity_fr.clone();
  global_output = r_lsbp.clone();
  part = 0;

  pthread_t threads[THREAD_MAX];

    for (int i = 0; i < THREAD_MAX; i++)
    pthread_create(&threads[i], NULL, LSBP1_parallel,
                            (void*)NULL);
    //quick_sort(aa.begin(), aa.end());

    for (int i = 0; i < THREAD_MAX; i++)
    pthread_join(threads[i], NULL);


  r_lsbp = global_output.clone();

  intensity_fr.release();
}

void* LSBP1_parallel(void* arg)
{
    int thread_part = part++;

  for(int k=1; k<((global_intensity_fr.rows-2)/THREAD_MAX)+1; k++)
  {
    int i = ((global_intensity_fr.rows-2)/THREAD_MAX)*thread_part+k;
    for(int j=1; j<global_intensity_fr.cols-1; j++)
    {
      arma::mat m_svd;
      m_svd = {{((Scalar)global_intensity_fr.at<uchar>(i-1,j-1))[0],
      ((Scalar)global_intensity_fr.at<uchar>(i-1,j))[0],
      ((Scalar)global_intensity_fr.at<uchar>(i-1,j+1))[0]},

      {((Scalar)global_intensity_fr.at<uchar>(i,j-1))[0],
      ((Scalar)global_intensity_fr.at<uchar>(i,j))[0],
      ((Scalar)global_intensity_fr.at<uchar>(i,j+1))[0]},

      {((Scalar)global_intensity_fr.at<uchar>(i+1,j-1))[0],
      ((Scalar)global_intensity_fr.at<uchar>(i+1,j))[0],
      ((Scalar)global_intensity_fr.at<uchar>(i+1,j+1))[0]}};

      global_output.at<float>(i,j) = _SVD(m_svd); 
    }
    
  }
 
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

void init_zeros_change_lsbp()
{
  Mat lsbp = Mat::zeros(width+2, heigth+2, CV_32FC1);
  list<Mat>::iterator next_lsbp;
  next_lsbp = samples_change_lsbp.begin();
  while(next_lsbp != samples_change_lsbp.end())
  {
    (*next_lsbp) = lsbp.clone();
    next_lsbp++;
  }
}

void init_change_lsbp()
{
  Mat lsbp = Mat::zeros(width+2, heigth+2, CV_32FC1);
  for(int i=0; i<10; i++)
  {
    samples_change_lsbp.push_back(lsbp);
  }
}

//threshold  HR PY
// matches   threshold PY
Mat SVD_step(Mat frame, int threshold=4, int matches=2, int Rscale=5, double Rlr=0.05, double Tlr=0.02)
{
  //Mat svd_fr = Mat::zeros(frame.rows+2, frame.cols+2, CV_32FC1);
  Mat svd_fr = Mat::zeros(width+2, heigth+2, CV_32FC1);

  //auto t11 = std::chrono::high_resolution_clock::now();
  //extract_LSBP(frame, svd_fr, 0.05);
  extract_LSBP_cuda(frame, svd_fr);
  //auto t12 = std::chrono::high_resolution_clock::now();
  //cout << "LSBP STEP: "<<std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t11).count() << endl;

  Mat mask = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
  
  //Mat white = Mat::ones(1,1, CV_8UC1)*255;
//#pragma omp parallell for
  
  
  auto t21 = std::chrono::high_resolution_clock::now();
  for(int i=0; i<frame.rows; i++)
  {
    for(int j=0; j<frame.cols; j++)
    {
      list<Mat>::iterator next_frame;
      next_frame = samples_frame.begin();

      list<Mat>::iterator next_lsbp;
      next_lsbp = samples_lsbp.begin();

      int samples_matches = 0;
      double min_distance_sum = 0;
      //double L1_distance_min = 1000000;

      while(next_lsbp != samples_lsbp.end())
      {
        double L1_distance = 
        get_distance_L1(((Scalar)frame.at<Vec3b>(i, j))[0], ((Scalar)(*next_frame).at<Vec3b>(i, j))[0],
          ((Scalar)frame.at<Vec3b>(i, j))[1], ((Scalar)(*next_frame).at<Vec3b>(i, j))[1],
          ((Scalar)frame.at<Vec3b>(i, j))[2], ((Scalar)(*next_frame).at<Vec3b>(i, j))[2]);

        double min_distance = 
        min_distance_L1(((Scalar)frame.at<Vec3b>(i, j))[0], ((Scalar)(*next_frame).at<Vec3b>(i, j))[0],
          ((Scalar)frame.at<Vec3b>(i, j))[1], ((Scalar)(*next_frame).at<Vec3b>(i, j))[1],
          ((Scalar)frame.at<Vec3b>(i, j))[2], ((Scalar)(*next_frame).at<Vec3b>(i, j))[2]);
        
        int d_hamming = Hamming_distance(svd_fr, *next_lsbp, i+1, j+1, 0.05);
    
        min_distance_sum += min_distance;
      
        //UPDATE R
        //D.at<float>(i,j) = L1_distance;

        
        //if((d_hamming < (threshold)))           
        if((L1_distance < R.at<float>(i, j)) && (d_hamming < (threshold)))            
        {
          samples_matches++;      
        }
        next_frame++;
        next_lsbp++;
      
      }

      //UPDATE R(x)
      if(R.at<float>(i, j) > ((min_distance_sum/10)*Rscale))
      {
        if(R.at<float>(i, j)*(1-Rlr) < 18)
          R.at<float>(i, j) = 18;
        else
          R.at<float>(i, j) = R.at<float>(i, j)*(1-Rlr);
      }
      else
      {
        if(R.at<float>(i, j)*(1+Rlr) > 500)
          R.at<float>(i, j) = 500;
        else
          R.at<float>(i, j) = R.at<float>(i, j)*(1+Rlr);
      }

      if(samples_matches < matches)
      {
        mask.at<uchar>(i, j) = 255;//White, Black 0

        //UPDATE T(X)
        if(T.at<float>(i, j)+(1/(min_distance_sum/10)) < 200)
          T.at<float>(i, j) = T.at<float>(i, j)+(1/(min_distance_sum/10));//IF FOREGROUND
        else
          T.at<float>(i, j) = 200;
      }
      else
      {
        //UPDATE T(X)
        if(T.at<float>(i, j)-(0.05/(min_distance_sum/10)) > 2)
          T.at<float>(i, j) = T.at<float>(i, j)-(0.05/(min_distance_sum/10));//IF BACKGROUND
        else
          T.at<float>(i, j) = 2;

        //UPDATE B(X) because mask(i,j) is 0
        if((rand()%200) < (200/T.at<float>(i, j)))
        {
          int random = rand()%10;
          
          list<Mat>::iterator next_frame_update;
          next_frame_update = samples_frame.begin();

          list<Mat>::iterator next_lsbp_update;
          next_lsbp_update = samples_lsbp.begin();

          for(int k=0; k<random; k++)
          {
            next_frame_update++;
            next_lsbp_update++;
 
          }
          (*next_frame_update).at<Vec3b>(i, j) = frame.at<Vec3b>(i,j);
  
        }
      }
    }
    
  } 
  
  auto t22 = std::chrono::high_resolution_clock::now();
  //cout << "COMPARE: "<<std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t21).count() << endl;
  update_samples_lsbp();
  
  init_zeros_change_lsbp();

  return mask;

}

/*Mat SVD_step(Mat frame, int threshold=4, int matches=2, int Rscale=5, double Rlr=0.05, double Tlr=0.02)
{
  //Mat svd_fr = Mat::zeros(frame.rows+2, frame.cols+2, CV_32FC1);
  Mat svd_fr = Mat::zeros(width+2, heigth+2, CV_32FC1);
  extract_LSBP(frame, svd_fr, 0.05);
  Mat mask = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
  
  //Mat white = Mat::ones(1,1, CV_8UC1)*255;
//#pragma omp parallell for
  global_frame = frame.clone();
  global_svd = svd_fr.clone();
  global_mask = mask.clone();
  part = 0;

  pthread_t threads[THREAD_MAX];

    for (int i = 0; i < THREAD_MAX; i++)
    pthread_create(&threads[i], NULL, SVD_step_parallell,
                            (void*)NULL);
    //quick_sort(aa.begin(), aa.end());

    for (int i = 0; i < THREAD_MAX; i++)
    pthread_join(threads[i], NULL);
    

  update_samples_lsbp();
  init_zeros_change_lsbp();

  mask = global_mask.clone();

  return mask;
}
*/
/*void* SVD_step_parallell(void * arg)
{
  int threshold = 6;
  int matches = 2;
  int Rscale = 5;
  double Rlr = 0.05;
  double Tlr = 0.02;

  for(int k=0; k<global_frame.rows/THREAD_MAX; k++)
  {
    int i=(global_frame.rows)/THREAD_MAX+k;
    for(int j=0; j<global_frame.cols; j++)
    {
      list<Mat>::iterator next_frame;
      next_frame = samples_frame.begin();

      list<Mat>::iterator next_lsbp;
      next_lsbp = samples_lsbp.begin();

      int samples_matches = 0;
      double min_distance_sum = 0;
      //double L1_distance_min = 1000000;

      while(next_lsbp != samples_lsbp.end())
      {

        double L1_distance = 
        get_distance_L1(((Scalar)global_frame.at<Vec3b>(i, j))[0], ((Scalar)(*next_frame).at<Vec3b>(i, j))[0],
          ((Scalar)global_frame.at<Vec3b>(i, j))[1], ((Scalar)(*next_frame).at<Vec3b>(i, j))[1],
          ((Scalar)global_frame.at<Vec3b>(i, j))[2], ((Scalar)(*next_frame).at<Vec3b>(i, j))[2]);

        double min_distance = 
        min_distance_L1(((Scalar)global_frame.at<Vec3b>(i, j))[0], ((Scalar)(*next_frame).at<Vec3b>(i, j))[0],
          ((Scalar)global_frame.at<Vec3b>(i, j))[1], ((Scalar)(*next_frame).at<Vec3b>(i, j))[1],
          ((Scalar)global_frame.at<Vec3b>(i, j))[2], ((Scalar)(*next_frame).at<Vec3b>(i, j))[2]);
        
        int d_hamming = Hamming_distance(global_svd, *next_lsbp, i+1, j+1, 0.05);
    
        min_distance_sum += min_distance;
      
        //UPDATE R
        D.at<float>(i,j) = L1_distance;

        
        //if((d_hamming < (threshold)))           
        if((L1_distance < R.at<float>(i, j)) && (d_hamming < (threshold)))            
        {
          samples_matches++;      
        }
        next_frame++;
        next_lsbp++;
      
      }

      //UPDATE R(x)
      if(R.at<float>(i, j) > ((min_distance_sum/10)*Rscale))
      {
        if(R.at<float>(i, j)*(1-Rlr) < 18)
          R.at<float>(i, j) = 18;
        else
          R.at<float>(i, j) = R.at<float>(i, j)*(1-Rlr);
      }
      else
      {
        if(R.at<float>(i, j)*(1+Rlr) > 500)
          R.at<float>(i, j) = 500;
        else
          R.at<float>(i, j) = R.at<float>(i, j)*(1+Rlr);
      }

      if(samples_matches < matches)
      {
        global_mask.at<uchar>(i, j) = 255;//White, Black 0

        //UPDATE T(X)
        if(T.at<uchar>(i, j)+(1/(min_distance_sum/10)) < 200)
          T.at<uchar>(i, j) = T.at<uchar>(i, j)+(1/(min_distance_sum/10));//IF FOREGROUND
        else
          T.at<uchar>(i, j) = 200;
      }
      else
      {
        //UPDATE T(X)
        if(T.at<uchar>(i, j)-(0.05/(min_distance_sum/10)) > 2)
          T.at<uchar>(i, j) = T.at<uchar>(i, j)-(0.05/(min_distance_sum/10));//IF BACKGROUND
        else
          T.at<uchar>(i, j) = 2;

        //UPDATE B(X) because mask(i,j) is 0
        if((rand()%200) < (200/T.at<uchar>(i, j)))
        {
          int random = rand()%10;
          
          list<Mat>::iterator next_frame_update;
          next_frame_update = samples_frame.begin();

          list<Mat>::iterator next_lsbp_update;
          next_lsbp_update = samples_lsbp.begin();

          list<Mat>::iterator next_change_lsbp_update;
          next_change_lsbp_update = samples_change_lsbp.begin();

          for(int k=0; k<random; k++)
          {
            next_frame_update++;
            next_lsbp_update++;
            next_change_lsbp_update++;
          }
          (*next_frame_update).at<Vec3b>(i, j) = global_frame.at<Vec3b>(i,j);
          (*next_change_lsbp_update).at<float>(i+1, j+1) = 1;
        }
      }
    }
    
  }
}*/

double get_distance_L1(double b1, double b2, double g1, double g2, double r1, double r2)
{
  return sqrt(pow((b1-b2),2)+pow((g1-g2),2)+pow((r1-r2),2));
}

double min_distance_L1(double b1, double b2, double g1, double g2, double r1, double r2)
{
  if(abs(b1-b2) < abs(g1-g2))
  {
    if(abs(r1-r2) < abs(b1-b2))
    {
      return abs(r1-r2);
    }
    else
      return abs(b1-b2);
  }
  else
  {
    if(abs(r1-r2) < abs(g1-g2))
    {
      return abs(r1-r2);
    }
    else
      return abs(g1-g2);
  }
}

/*void update_samples_lsbp()
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
*/

/*void extract_LSBP_v2(Mat frame, Mat frame_change, Mat last_lsbp,Mat &r_lsbp, int tau=0.05)
{
  Mat intensity;
  cvtColor(frame, intensity, COLOR_BGR2GRAY);
  Mat intensity_fr = Mat::zeros(frame.rows+2, frame.cols+2, CV_8UC1);

  for(int i=1; i<intensity_fr.rows-1; i++)
  {
    for(int j=1; j<intensity_fr.cols-1; j++)
    {
      intensity_fr.at<uchar>(i,j) = intensity.at<uchar>(i-1,j-1); 
    }
  }

  for(int i=1; i<intensity_fr.rows-1; i++)
  {
    for(int j=1; j<intensity_fr.cols-1; j++)
    {
      if(validate_change(frame_change.at<float>(i-1,j-1), frame_change.at<float>(i-1,j), frame_change.at<float>(i-1,j+1),
        frame_change.at<float>(i,j-1), frame_change.at<float>(i,j), frame_change.at<float>(i,j+1),
        frame_change.at<float>(i+1,j-1), frame_change.at<float>(i+1,j), frame_change.at<float>(i+1,j+1)))
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
      }
      else
      {
        r_lsbp.at<float>(i,j) = last_lsbp.at<float>(i,j);
      }
    } 
    
  }

  intensity_fr.release();
  
}*/

void extract_LSBP_v2(Mat frame, Mat frame_change, Mat last_lsbp,Mat &r_lsbp, int tau=0.05)
{
  Mat intensity;
  cvtColor(frame, intensity, COLOR_BGR2GRAY);
  Mat intensity_fr = Mat::zeros(frame.rows+2, frame.cols+2, CV_8UC1);

  for(int i=1; i<intensity_fr.rows-1; i++)
  {
    for(int j=1; j<intensity_fr.cols-1; j++)
    {
      intensity_fr.at<uchar>(i,j) = intensity.at<uchar>(i-1,j-1); 
    }
  }

  part=0;
  global_intensity_fr = intensity_fr.clone();
  global_change_frame = frame_change.clone();
  global_last_lsbp = last_lsbp.clone();
  global_output = r_lsbp.clone();

  pthread_t threads[THREAD_MAX];

    for (int i = 0; i < THREAD_MAX; i++)
    pthread_create(&threads[i], NULL, LSBP_parallel,
                            (void*)NULL);
    //quick_sort(aa.begin(), aa.end());

    for (int i = 0; i < THREAD_MAX; i++)
    pthread_join(threads[i], NULL);

  r_lsbp = global_output.clone();

  intensity_fr.release();
  
}

void* LSBP_parallel(void* arg)
{
    int thread_part = part++;
  
    /*for(int i=0; i<global_mat.rows/THREAD_MAX; i++)
    {
        for(int j=0; j<global_mat.cols; j++)
        {
            global_mat.at<float>((global_mat.rows/THREAD_MAX)*thread_part+i,j) = 1;
        }
    }*/
   
    for(int k=1; k<((global_intensity_fr.rows-2)/THREAD_MAX)+1; k++)
  {
    int i = ((global_intensity_fr.rows-2)/THREAD_MAX)*thread_part+k;
    //cout << "thread: " << thread_part <<" i: " << i << endl;
    for(int j=1; j<global_intensity_fr.cols-1; j++)
    {
      if(validate_change(global_change_frame.at<float>(i-1,j-1), global_change_frame.at<float>(i-1,j), global_change_frame.at<float>(i-1,j+1),
        global_change_frame.at<float>(i,j-1), global_change_frame.at<float>(i,j), global_change_frame.at<float>(i,j+1),
        global_change_frame.at<float>(i+1,j-1), global_change_frame.at<float>(i+1,j), global_change_frame.at<float>(i+1,j+1)))
      {
        arma::mat m_svd;
        m_svd = {{((Scalar)global_intensity_fr.at<uchar>(i-1,j-1))[0],
        ((Scalar)global_intensity_fr.at<uchar>(i-1,j))[0],
        ((Scalar)global_intensity_fr.at<uchar>(i-1,j+1))[0]},

        {((Scalar)global_intensity_fr.at<uchar>(i,j-1))[0],
        ((Scalar)global_intensity_fr.at<uchar>(i,j))[0],
        ((Scalar)global_intensity_fr.at<uchar>(i,j+1))[0]},

        {((Scalar)global_intensity_fr.at<uchar>(i+1,j-1))[0],
        ((Scalar)global_intensity_fr.at<uchar>(i+1,j))[0],
        ((Scalar)global_intensity_fr.at<uchar>(i+1,j+1))[0]}};

        global_output.at<float>(i,j) = _SVD(m_svd);
      }
      else
      {
        global_output.at<float>(i,j) = global_last_lsbp.at<float>(i,j);
      }
    } 
    
  }
 
}

bool validate_change(float i00, float i01, float i02, float i10, float i11, float i12, float i20, float i21, float i22)
{
  if(i00 == 1 || i01 == 1 ||i02 == 1 ||i10 == 1 ||i11 == 1 ||i12 == 1 ||i20 == 1 ||i21 == 1 ||i22 == 1)
    return true;
  else
    return false;
}

void update_samples_lsbp()
{
  list<Mat>::iterator next_frame;
  next_frame = samples_frame.begin();

  list<Mat>::iterator next_lsbp;
  next_lsbp = samples_lsbp.begin();

  list<Mat>::iterator next_lsbp_change;
  next_lsbp_change = samples_change_lsbp.begin();

  //int samples_matches = 0;

  while(next_lsbp != samples_lsbp.end())
  {
    Mat svd = Mat::zeros(width+2, heigth+2, CV_32FC1);
    //extract_LSBP_v2(*next_frame, *next_lsbp_change, *next_lsbp,svd, 0.05);
    extract_LSBP_cuda(*next_frame, svd);
    //extract_LSBP(*next_frame, svd, 0.05);
    *next_lsbp = svd.clone();

    next_lsbp_change++;
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

void extract_LSBP_cuda(Mat frame, Mat &output)
{
  Mat intensity;
  cvtColor(frame, intensity, COLOR_BGR2GRAY);
  //Mat intensity_fr = Mat::zeros(frame.rows+2, frame.cols+2, CV_8UC1);

  /*for(int i=1; i<intensity_fr.rows-1; i++)
  {
    for(int j=1; j<intensity_fr.cols-1; j++)
    {
      intensity_fr.at<uchar>(i,j) = intensity.at<uchar>(i-1,j-1); 
    }
  }*/

  //int filas = 1080;
  //int columnas = 1920;
  int filas = frame.rows+2;
  int columnas = frame.cols+2;
  
  float a[filas][columnas];
 
  float *dev_a;
  float *dev_c;

  for(int i=1; i<filas-1; i++)
  {
    //cont = 0;
    for(int j=1; j<columnas-1; j++)
    {
      a[i][j] = ((Scalar)intensity.at<uchar>(i-1, j-1))[0];
    }
  }

 // auto t11 = std::chrono::high_resolution_clock::now();

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

  //auto t12 = std::chrono::high_resolution_clock::now();

  cudaFree(dev_a);
  cudaFree(dev_c);

  for(int i=0; i<filas; i++)
  {
    for(int j=0; j<columnas; j++)
    {
      ((Scalar)output.at<uchar>(i,j))[0] = a[i][j];
    }
  }
  
  //cout << std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t11).count() << endl;
}