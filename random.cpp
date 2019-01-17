#include <iostream>
#include <time.h>

using namespace std;

int clip(int i, int inferior, int superior, int val_range);

int main()
{
	srand(time(NULL));
	int i0, j0;
	for(int i=0; i< 100; i++)
	{
		i0 = clip(i,10,90,10);
		cout << "-> "<< i0 << endl;
	}

	for(int j=0; j< 100; j++)
	{
		j0 = clip(j,10,90,10);
	}
	cout << rand()%10 << endl;
	return 0;
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