#include <iostream>
#include <string.h>
#include <experimental/filesystem>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

int main()
{
	srand(time(NULL));
	for(int i=0; i<10; i++)
		cout << rand()%10+1<< endl;
	return 0;
	
}