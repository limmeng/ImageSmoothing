#ifndef _IMAGE_SMOOTHER_H
#define _IMAGE_SMOOTHER_H

#include <cmath>
#include <opencv2/core/core.hpp>
using namespace cv;

class ImageSmoother {
public:
	ImageSmoother(){}
	~ImageSmoother(){}

	void smooth(const Mat srcImage, Mat &desImage, const double lambda, const double kappad);

private:
	void getHVfromImage(const Mat image, Mat* h, Mat* v, const double boundary);
	void getImageFromHV(Mat Denormin2,  Mat *Normin1, Mat &des, Mat* h, Mat* v, const double beta, const double lambda);
	
	void gradientDecent(const Mat src, Mat &des, Mat h, Mat v, const double beta, const double lambda);

	void shiftMatrix(Mat &mat, int shiftX, int shiftY);
	Mat psf2otf(const Mat &psf, const Size &outSize);
	int deltaOfTwo(Vec3b x, Vec3b y);
	bool gradientDecentLoop(int oldNum, int newNum);
	void updateSumMatrix(const Mat src, const Mat des, Mat &temp, const Mat h, const Mat v);

	static const double bataMax;
};

#endif // !_IMAGE_SMOOTHER_H
