#include "siltp.h"
#include <functional>

void SILTP_8x1(InputArray _src, OutputArray _dst)
{
	Mat src = _src.getMat();

	cvtColor(src, src, COLOR_BGR2GRAY);
	Mat texture = Mat::zeros(src.rows - 2, src.cols - 2, CV_16UC1);

	auto compare = [](uint8_t center, uint8_t neighbor) -> uint8_t
	{
		float tau = 0.025;

		if (neighbor > (1.0 + tau)*center)
			return 1;
		else if (neighbor < (1.0 - tau)*center)
			return 2;
		else
			return 0;
	};

	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			uint8_t center = src.at<uint8_t>(i, j);
			uint16_t code = 0;
			code |= compare(center, src.at<uint8_t>(i-1,j-1)) << 15;
			code |= compare(center, src.at<uint8_t>(i-1,j)) << 13;
			code |= compare(center, src.at<uint8_t>(i-1,j+1)) << 11;
			code |= compare(center, src.at<uint8_t>(i,j+1)) << 9;
			code |= compare(center, src.at<uint8_t>(i+1,j+1)) << 7;
			code |= compare(center, src.at<uint8_t>(i+1,j)) << 5;
			code |= compare(center, src.at<uint8_t>(i+1,j-1)) << 3;
			code |= compare(center, src.at<uint8_t>(i,j-1)) << 1;
			
			texture.at<uint16_t>(i-1,j-1) = code;
		}
	}

	texture.copyTo(_dst);
}

void SILTP_16x2(InputArray _src, OutputArray _dst)
{
	Mat src = _src.getMat();

	cvtColor(src, src, COLOR_BGR2GRAY);
	Mat texture = Mat::zeros(src.rows - 4, src.cols - 4, CV_32S);

	auto compare = [](uint8_t center, uint8_t neighbor) -> uint8_t
	{
		float tau = 0.04;

		if (neighbor > (1.0 + tau)*center)
			return 1;
		else if (neighbor < (1.0 - tau)*center)
			return 2;
		else
			return 0;
	};

	for (int i = 2; i < src.rows - 2; i++)
	{
		for (int j = 2; j < src.cols - 2; j++)
		{
			uint8_t center = src.at<uint8_t>(i, j);
			uint32_t code = 0;
			code |= compare(center, src.at<uint8_t>(i-2,j-2)) << 31;
			code |= compare(center, src.at<uint8_t>(i-2,j-1)) << 29;
			code |= compare(center, src.at<uint8_t>(i-2,j)) << 27;
			code |= compare(center, src.at<uint8_t>(i-2,j+1)) << 25;
			code |= compare(center, src.at<uint8_t>(i-2,j+2)) << 23;
			code |= compare(center, src.at<uint8_t>(i-1,j+2)) << 21;
			code |= compare(center, src.at<uint8_t>(i,j+2)) << 19;
			code |= compare(center, src.at<uint8_t>(i+1,j+2)) << 17;
			code |= compare(center, src.at<uint8_t>(i+2,j+2)) << 15;
			code |= compare(center, src.at<uint8_t>(i+2,j+1)) << 13;
			code |= compare(center, src.at<uint8_t>(i+2,j)) << 11;
			code |= compare(center, src.at<uint8_t>(i+2,j-1)) << 9;
			code |= compare(center, src.at<uint8_t>(i+2,j-2)) << 7;
			code |= compare(center, src.at<uint8_t>(i+1,j-2)) << 5;
			code |= compare(center, src.at<uint8_t>(i,j-2)) << 3;
			code |= compare(center, src.at<uint8_t>(i-1,j-2)) << 1;
			
			texture.at<uint32_t>(i-2,j-2) = code;
		}
	}

	texture.copyTo(_dst);

}

uint8_t hamming_distance(uint32_t a, uint32_t b)
{
	uint8_t dist = 0;
	unsigned  val = a ^ b;
	
	// Count the number of bits set
	while (val != 0)
	{
		// A bit is set, so increment the count and clear the bit
		dist++;
		val &= val - 1;
	}
   	// Return the number of differing bits
	return dist;
}
