#include <algorithm>
#include <opencv2/opencv.hpp>
#include "utils.h"

float argmax(const std::vector<std::tuple<float, uint32_t>>& Q)
{
	std::vector<float> psi_L;
	psi_L.resize(Q.size());
	std::transform(Q.begin(), Q.end(), psi_L.begin(),
			[](const std::tuple<float, uint32_t>& A) { return std::get<0>(A); });

	cv::Mat tmp(psi_L, false);

	int bins = 1024;
	float range[] = { 0, 255 };
	const float* histRange = { range };

	cv::Mat L_hist;
	cv::calcHist(&tmp, 1, 0, cv::Mat(), L_hist, 1, &bins, &histRange, true, false);

	float max = 0;
	float argmax = 0;
	for (int i = 0; i < bins; i++)
	{
		float value = L_hist.at<float>(i);
		if (value > max)
		{
			max = value;
			argmax = (i/float(bins)) * 255;
		}
	}

//	int hist_w = 512; int hist_h = 400;
//	int bin_w = cvRound( (double) hist_w/bins);
//	cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
//	for( int i = 1; i < bins; i++ )
//	{
//		line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(L_hist.at<float>(i-1)) ) ,
//				cv::Point( bin_w*(i), hist_h - cvRound(L_hist.at<float>(i)) ),
//				cv::Scalar( 255, 0, 0), 2, 8, 0  );
//	}
//	std::cout << "mode = " << mode << std::endl;
//
//	cv::namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
//	  imshow("calcHist Demo", histImage );
//
//	  cv::waitKey(0);

	return argmax;
}
