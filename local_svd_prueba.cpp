#include <iostream>
#include <utility>
#include <stdlib.h>
#include <time.h>
#include "vector"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <string.h>
#include <fstream>
#include <chrono>
 
// number of elements in array
 
// number of threads
#define THREAD_MAX 4

using namespace std;
using namespace cv;

int part = 0;
Mat global_mat;
//template<class RandomAccessIterator>
void* quick_sort(void* arg);
void export_mat_excel(Mat img, string name);

int main() {

    auto t11 = std::chrono::high_resolution_clock::now();

    Mat init = Mat::zeros(3000, 1000, CV_32FC1);
    global_mat = init.clone();

    pthread_t threads[THREAD_MAX];

    for (int i = 0; i < THREAD_MAX; i++)
    pthread_create(&threads[i], NULL, quick_sort,
                            (void*)NULL);
    //quick_sort(aa.begin(), aa.end());

    for (int i = 0; i < THREAD_MAX; i++)
    pthread_join(threads[i], NULL);

    auto t12 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t11).count();
    cout << "Time of proccess: " << duration << endl;

    export_mat_excel(global_mat, "global_mat");

    return 0;
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

void* quick_sort(void* arg)
{
    int thread_part = part++;
    cout << thread_part << endl;

    for(int i=0; i<global_mat.rows/THREAD_MAX; i++)
    {
        for(int j=0; j<global_mat.cols; j++)
        {
            global_mat.at<float>((global_mat.rows/THREAD_MAX)*thread_part+i,j) = 1;
        }
    }
    //cout << thread_part * ((aa.end()-aa.begin()) / 4) << "->" << (thread_part + 1) * ((aa.end()-aa.begin()) / 4) << endl;

    //cout << aa.end()-aa.begin() << endl;
    //merge_sort(aa.begin()+(thread_part * ((aa.end()-aa.begin()) / 4)), aa.begin()+((thread_part + 1) * ((aa.end()-aa.begin()) / 4) ));
 
}


       