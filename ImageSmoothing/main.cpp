#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "ImageSmoother.h"

using namespace cv;
using namespace std;
int main() 
{
	string fileName = "C:\\Users\\limeng\\Desktop\\test.jpg";
	Mat image;
	image = imread(fileName, CV_LOAD_IMAGE_COLOR);

	if(!image.data) 
	{
		cout << "Could not open image file." << endl;
		system("pause");
		return -1;
	}

	namedWindow("display window", WINDOW_AUTOSIZE);
	imshow("display window", image);

	ImageSmoother smoother;
	Mat smoothImage;
	smoother.smooth(image, smoothImage, 0.01);

	if(!smoothImage.data) 
	{
		cout << "Could not open smoothImage." << endl;
		system("pause");
		return -1;
	}

	namedWindow("smooth window", WINDOW_AUTOSIZE);
	imshow("smooth window", smoothImage);

	waitKey(0);
	return 0;
}
