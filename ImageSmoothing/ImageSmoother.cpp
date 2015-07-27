#include "ImageSmoother.h"

const double ImageSmoother::cBMax = 1E5;

void ImageSmoother::smooth(const Mat srcImage, Mat &desImage, const double s) {
	double b = 2 * s;
	desImage = srcImage.clone();
	Mat h = Mat::zeros(desImage.size(), CV_16U), 
		v = Mat::zeros(desImage.size(), CV_16U);

	while (b < cBMax)
	{
		getHVfromImage(desImage, h, v, s/b);
		getImageFromHV(srcImage, desImage, h, v, b, s);
		b *= cRate;
	}
	
	return;
}

void ImageSmoother::getHVfromImage(const Mat image, Mat &h, Mat &v, const double boundary)
{
	for (int i = 0, iCount = image.rows - 1; i < iCount; ++i) 
	{
		for (int j = 0, jCount = image.cols - 1; j < jCount; ++j)
		{
			int deltaX = deltaOfTwo(image.at<Vec3b>(i, j), image.at<Vec3b>(i, j+1));
			int deltaY = deltaOfTwo(image.at<Vec3b>(i, j), image.at<Vec3b>(i+1, j));

			if (pow(deltaX, 2) + pow(deltaY, 2) <= boundary)
			{
				h.at<unsigned short>(i, j) = 0;
				v.at<unsigned short>(i, j) = 0;
			}
			else 
			{
				h.at<unsigned short>(i, j) = deltaX;
				v.at<unsigned short>(i, j) = deltaY;
			}
		}
	}
}

void ImageSmoother::getImageFromHV(const Mat src, Mat &des, Mat h, Mat v, const double b, const double s) 
{
	des = src.clone();
	Mat temp = Mat::zeros(des.size(), CV_64F);
	
	double *p;
	unsigned short *hp, *vp;
	for (int i = 0, iCount = temp.rows - 1; i < iCount; ++i)
	{
		p = temp.ptr<double>(i);
		hp = h.ptr<unsigned short>(i);
		vp = v.ptr<unsigned short>(i);
		
		for (int j = 0, jCount = temp.cols - 1; j < jCount; ++j)
		{
			int deltaX = deltaOfTwo(des.at<Vec3b>(i, j), des.at<Vec3b>(i, j+1));
			int deltaY = deltaOfTwo(des.at<Vec3b>(i, j), des.at<Vec3b>(i+1, j));
			// sum format
			p[j] = pow(deltaOfTwo(des.at<Vec3b>(i, j), src.at<Vec3b>(i, j)), 2) + b * (pow(deltaX - hp[j], 2) + pow(deltaY - vp[j], 2));
			if (abs(hp[j]) + abs(vp[j]) != 0) 
			{
				p[j] += s;
			}
		}
	}
	double sumNum = sum(temp)[0];

	double lastSumNum;
	int count = 0;
	do 
	{
		lastSumNum = sumNum;

		// update result Mat
		for (int i = 0, iCount = des.rows - 1; i < iCount; ++i) 
		{
			for (int j = 0, jCount = des.cols - 1; j < jCount; ++j)
			{
				int deltaX = deltaOfTwo(des.at<Vec3b>(i, j), des.at<Vec3b>(i, j+1));
				int deltaY = deltaOfTwo(des.at<Vec3b>(i, j), des.at<Vec3b>(i+1, j));
				// derivative format
				// TODO 
				double derivative = 2 * (deltaOfTwo(src.at<Vec3b>(i, j), des.at<Vec3b>(i, j+1)) 
					+ b * (deltaX + deltaY - h.at<unsigned short>(i, j) - v.at<unsigned short>(i, j)));

				des.at<Vec3b>(i, j) = des.at<Vec3b>(i, j) - des.at<Vec3b>(i, j) * derivative * 0.01;
			}
		}

		// update sum number
		for (int i = 0, iCount = temp.rows - 1; i < iCount; ++i)
		{
			p = temp.ptr<double>(i);
			hp = h.ptr<unsigned short>(i);
			vp = v.ptr<unsigned short>(i);

			for (int j = 0, jCount = temp.cols - 1; j < jCount; ++j)
			{
				int deltaX = deltaOfTwo(des.at<Vec3b>(i, j), des.at<Vec3b>(i, j+1));
				int deltaY = deltaOfTwo(des.at<Vec3b>(i, j), des.at<Vec3b>(i+1, j));
				p[j] = pow(deltaOfTwo(des.at<Vec3b>(i, j), src.at<Vec3b>(i, j)), 2) + b * (pow(abs(deltaX - hp[j]), 2) + pow(abs(deltaY - vp[j]), 2));
				if (abs(hp[j]) + abs(vp[j]) != 0) 
				{
					p[j] += s;
				}
			}
		}
		sumNum = sum(temp)[0];
		count++;
	} while (count != 10);//gradientDecentLoop(lastSumNum, sumNum));
}

// 返回图像中两个像素点之间rgb值的差的绝对值之和
int ImageSmoother::deltaOfTwo(Vec3b x, Vec3b y)
{
	return abs(x.val[0] - y.val[0]) + abs(x.val[1] - y.val[1]) + abs(x.val[2] - y.val[2]);
}

bool ImageSmoother::gradientDecentLoop(int oldNum, int newNum) 
{
	double accuracy = 0.1;
	return (abs(newNum-oldNum) < accuracy * oldNum ? false : true);
}