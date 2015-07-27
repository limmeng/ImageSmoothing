#ifndef _IMAGE_SMOOTHER_H
#define _IMAGE_SMOOTHER_H

#include <cmath>
#include <opencv2/core/core.hpp>
using namespace cv;

class ImageSmoother {
public:
	ImageSmoother(){}
	~ImageSmoother(){}

	void smooth(const Mat srcImage, Mat &desImage, const double s);

private:
	void getHVfromImage(const Mat image, Mat &h, Mat &v, const double boundary);
	void getImageFromHV(const Mat src, Mat &des, Mat h, Mat v, const double b, const double s);
	
	int deltaOfTwo(Vec3b x, Vec3b y);
	bool gradientDecentLoop(int oldNum, int newNum);

	static const double cBMax;
	static const int cRate = 2;
};

#endif // !_IMAGE_SMOOTHER_H
