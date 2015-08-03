#include "ImageSmoother.h"

const double ImageSmoother::bataMax = 1E5;

// Description: smooth image.
void ImageSmoother::smooth(const Mat srcImage, Mat &desImage, const double lambda = 2e-2, const double kappad = 2.0) 
{
	srcImage.convertTo(desImage, CV_64FC3, 1.0/255.0);
	Mat h[3], v[3];

	// initial fft algorithm.
	int row = desImage.rows, col = desImage.cols;
	Mat Normin1[3];
	Mat single_channel[3];
	split(desImage, single_channel);
	for (int k = 0; k < 3; k++) 
	{
		dft(single_channel[k], Normin1[k], DFT_COMPLEX_OUTPUT);
	}

	Mat fx(1,2,CV_64FC1);
	Mat fy(2,1,CV_64FC1);
	fx.at<double>(0) = 1; fx.at<double>(1) = -1;
	fy.at<double>(0) = 1; fy.at<double>(1) = -1;
	Size sizeI2D = srcImage.size();
	Mat otfFx = psf2otf(fx, sizeI2D);
	Mat otfFy = psf2otf(fy, sizeI2D);
	Mat Denormin2(row, col, CV_64FC1);
	for (int i = 0; i < row; i++) 
	{
		for (int j = 0; j < col; j++) 
		{
			Vec2d &c1 = otfFx.at<Vec2d>(i,j), &c2 = otfFy.at<Vec2d>(i,j);
			Denormin2.at<double>(i,j) = pow(c1[0], 2) + pow(c1[1], 2) + pow(c2[0], 2) + pow(c2[1], 2);
		}
	}

	double beta = 2 * lambda;
	while (beta < bataMax)
	{
		getHVfromImage(desImage, h, v, lambda/beta);
		getImageFromHV(Denormin2, Normin1, desImage, h, v, beta, lambda);
		beta *= kappad;
	}
}

// Description: get H V Matrix from image.
void ImageSmoother::getHVfromImage(const Mat image, Mat *h, Mat *v, const double boundary)
{
	Mat single[3];
	split(image, single);
	for (int k = 0; k < 3; ++k)
	{
		Mat shiftXMat = single[k].clone();
		Mat shiftYMat = single[k].clone();

		shiftMatrix(shiftXMat, 0, -1);
		shiftMatrix(shiftYMat, -1, 0);

		h[k] = shiftXMat - single[k];
		v[k] = shiftYMat - single[k];
	}
	for (int i = 0, iCount = image.rows; i < iCount; ++i) 
	{
		for (int j = 0, jCount = image.cols; j < jCount; ++j)
		{
			double value = pow(h[0].at<double>(i, j), 2) + pow(v[0].at<double>(i, j), 2) + 
				pow(h[1].at<double>(i, j), 2) + pow(v[1].at<double>(i, j), 2) + 
				pow(h[2].at<double>(i, j), 2) + pow(v[2].at<double>(i, j), 2);

			if (value <= boundary)
			{
				h[0].at<double>(i, j) = h[1].at<double>(i, j) = h[2].at<double>(i, j) = 0.0;
				v[0].at<double>(i, j) = v[1].at<double>(i, j) = v[2].at<double>(i, j) = 0.0;
			}
		}
	}
}

// Description: get image from h v matrix, this is the main part of the algorithm.
void ImageSmoother::getImageFromHV(Mat Denormin2, Mat *Normin1, Mat &des, Mat *h, Mat *v, const double beta, const double lambda) 
{
	// gradientDecent(src, des, *h, *v, beta, lambda);
	Mat Denormin = 1.0 + beta*Denormin2;
	int row = des.rows, col = des.cols;
	Mat single[3];
	split(des, single);

	for (int k = 0; k < 3; ++k)
	{
		Mat shiftXMat = h[k].clone();
		Mat shiftYMat = v[k].clone();

		shiftMatrix(shiftXMat, 0, 1);
		shiftMatrix(shiftYMat, 1, 0);

		Mat ddx = shiftXMat - h[k];
		Mat ddy = shiftYMat - v[k];

		Mat Normin2 = ddx + ddy;
		Mat FNormin2;
		dft(Normin2, FNormin2, DFT_COMPLEX_OUTPUT);
		Mat FS = Normin1[k] + beta*FNormin2;
		for (int i = 0; i < row; i++) 
		{
			for (int j = 0; j < col; j++) 
			{
				FS.at<Vec2d>(i,j)[0] /= Denormin.at<double>(i,j);
				FS.at<Vec2d>(i,j)[1] /= Denormin.at<double>(i,j);
			}
		}
		Mat ifft;
		idft(FS, ifft, DFT_SCALE | DFT_COMPLEX_OUTPUT);
		for (int i = 0; i < row; i++) 
		{
			for (int j = 0; j < col; j++) 
			{
				single[k].at<double>(i,j) = ifft.at<Vec2d>(i,j)[0];
			}
		}
	}
	merge(single, 3, des);
}

// Description: using gradient decent algorithm.
void ImageSmoother::gradientDecent(const Mat src, Mat &des, Mat h, Mat v, const double beta, const double lambda) 
{
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
			p[j] = pow(deltaOfTwo(des.at<Vec3b>(i, j), src.at<Vec3b>(i, j)), 2) + beta * (pow(deltaX - hp[j], 2) + pow(deltaY - vp[j], 2));
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
				double derivative;
				if (0 == i && 0 == j) 
				{
					derivative = 2 * (deltaOfTwo(src.at<Vec3b>(i, j), des.at<Vec3b>(i, j+1)) + 
						beta * (deltaX + deltaY - h.at<unsigned short>(i, j) - v.at<unsigned short>(i, j)));
				}
				else if (0 == i && j != 0) 
				{
					int deltaX2 = deltaOfTwo(des.at<Vec3b>(i, j), des.at<Vec3b>(i, j-1));
					derivative = 2 * (deltaOfTwo(src.at<Vec3b>(i, j), des.at<Vec3b>(i, j+1)) + 
						beta * (deltaX + deltaY + deltaX2 - h.at<unsigned short>(i, j) - v.at<unsigned short>(i, j) - v.at<unsigned short>(i, j-1)));
				}
				else if (0 == j && i != 0) 
				{
					int deltaY2 = deltaOfTwo(des.at<Vec3b>(i, j), des.at<Vec3b>(i-1, j));
					derivative = 2 * (deltaOfTwo(src.at<Vec3b>(i, j), des.at<Vec3b>(i, j+1)) + 
						beta * (deltaX + deltaY + deltaY2 - h.at<unsigned short>(i, j) - v.at<unsigned short>(i, j) - v.at<unsigned short>(i-1, j)));
				}
				else 
				{
					int deltaX2 = deltaOfTwo(des.at<Vec3b>(i, j), des.at<Vec3b>(i, j-1));
					int deltaY2 = deltaOfTwo(des.at<Vec3b>(i, j), des.at<Vec3b>(i-1, j));
					derivative = 2 * (deltaOfTwo(src.at<Vec3b>(i, j), des.at<Vec3b>(i, j+1)) + 
						beta * (deltaX + deltaY + deltaY2 + deltaX2 
						- h.at<unsigned short>(i, j) - v.at<unsigned short>(i, j) - v.at<unsigned short>(i, j-1) - v.at<unsigned short>(i-1, j)));
				}

				des.at<Vec3b>(i, j) -= des.at<Vec3b>(i, j) * derivative * 0.01;
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
				p[j] = pow(deltaOfTwo(des.at<Vec3b>(i, j), src.at<Vec3b>(i, j)), 2) + beta * (pow(abs(deltaX - hp[j]), 2) + pow(abs(deltaY - vp[j]), 2));
			}
		}
		sumNum = sum(temp)[0];
		count++;
	} while (count != 10);//gradientDecentLoop(lastSumNum, sumNum));
}

void ImageSmoother::shiftMatrix(Mat &mat, int shiftX, int shiftY) 
{
	int row = mat.rows, col = mat.cols;
	shiftX = (row + (shiftX % row)) % row;
	shiftY = (col + (shiftY % col)) % col;

	Mat temp =  mat.clone();
	if (shiftX)
	{
		temp.rowRange(row - shiftX, row).copyTo(mat.rowRange(0, shiftX));
		temp.rowRange(0, row - shiftX).copyTo(mat.rowRange(shiftX, row));
	}
	if (shiftY)
	{
		temp.colRange(col - shiftY, col).copyTo(mat.colRange(0, shiftY));
		temp.colRange(0, col - shiftY).copyTo(mat.colRange(shiftY, col));
	}
	return;
}

// Description: return the sum of every distance of two pixels in RGB mode.
int ImageSmoother::deltaOfTwo(Vec3b x, Vec3b y)
{
	return abs(x.val[0] - y.val[0]) + abs(x.val[1] - y.val[1]) + abs(x.val[2] - y.val[2]);
}

// Description: whether gradient decent algorithm needs to continue.
bool ImageSmoother::gradientDecentLoop(int oldNum, int newNum) 
{
	double accuracy = 0.1;
	return (abs(newNum-oldNum) < accuracy * oldNum ? false : true);
}

void ImageSmoother::updateSumMatrix(const Mat src, const Mat des, Mat &temp, const Mat h, const Mat v) 
{
}

Mat ImageSmoother::psf2otf(const Mat &psf, const Size &outSize) {
	Size psfSize = psf.size();
	Mat new_psf = Mat(outSize, CV_64FC2);
	new_psf.setTo(0);

	for (int i = 0; i < psfSize.height; i++) 
	{
		for (int j = 0; j < psfSize.width; j++) 
		{
			new_psf.at<Vec2d>(i,j)[0] = psf.at<double>(i,j);
		}
	}

	shiftMatrix(new_psf, -1*int(floor(psfSize.height*0.5)), -1*int(floor(psfSize.width*0.5)));

	Mat otf;
	dft(new_psf, otf, DFT_COMPLEX_OUTPUT);

	return otf;
}