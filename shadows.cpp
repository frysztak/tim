#include "shadows.h"
#include "siltp.h"
#include "gco/GCoptimization.h"

void Shadows::Init(StaufferGrimson* gmm)
{
	this->gmm = gmm;
}

void Shadows::RemoveShadows(InputArray _src, InputArray _bg, OutputArray _dst)
{
	Mat frame = _src.getMat();
	Mat bg = _bg.getMat();

	Mat frameSILTP, bgSILTP, shadowMask, shadowMask2;
	SILTP_16x2(frame, frameSILTP);
	fct.setFgTexture(frameSILTP);
	shadowMask = Mat::zeros(frameSILTP.size(), CV_8U);
	shadowMask2 = Mat::zeros(frame.size(), CV_8U);

	//cv::FileStorage fs("fgTexture.yml", cv::FileStorage::WRITE);
	//fs << "fgTexture" << frameSILTP;

	for (int i = 2; i < frame.rows - 2; i++)
	{
		for (int j = 2; j < frame.cols - 2; j++)
		{
			uint8_t distance = hamming_distance(gmm->BackgroundTexture.at<uint32_t>(i-2, j-2), 
					frameSILTP.at<uint32_t>(i-2, j-2));
			uint8_t bg = gmm->Background.at<Colour>(i, j).x; // L
			uint8_t fg = frame.at<Colour>(i, j).x; // L
			if (distance < thresh1)// || frameSILTP.at<uint32_t>(i-2, j-2) < thresh2)
			{
				if (fg/float(bg) <= 0.9)
				{
					shadowMask.at<uint8_t>(i-2, j-2) = 1;
				}
			}
		}
	}

	//medianBlur(shadowMask, shadowMask, 3);
	
	// detect shadows in colour space
	//Mat bgHSV, fgHSV;
	//cvtColor(frame, fgHSV, COLOR_BGR2HSV);
	//cvtColor(gmm->Background, bgHSV, COLOR_BGR2HSV);

	//std::vector<Mat> fg_channels, bg_channels;
	//split(fgHSV, fg_channels);
	//split(bgHSV, bg_channels);

	//Mat R, D_H, D_S;
	//R = Mat::zeros(frame.size(), CV_32F);
	//D_H = Mat::zeros(frame.size(), CV_32F);
	//D_S = Mat::zeros(frame.size(), CV_32F);
	//subtract(fg_channels[0], bg_channels[0], D_H, noArray(), CV_32F); // H
	//subtract(fg_channels[1], bg_channels[1], D_S, noArray(), CV_32F); // S
	//divide(fg_channels[2], bg_channels[2], R, 1, CV_32F); // V

	////double min, max;
	////minMaxLoc(D_H, &min, &max);
	////D_H /= max;
	////imshow("dum dum", D_H);

	//Scalar mean_H, stdDev_H;
	//Scalar mean_S, stdDev_S;
	//Scalar mean_V, stdDev_V;
	//meanStdDev(D_H, mean_H, stdDev_H, shadowMask);
	//meanStdDev(D_S, mean_S, stdDev_S, shadowMask);
	//meanStdDev(R, mean_V, stdDev_V, shadowMask);

	//const double a = 6;
	//float beta1 = mean_V[0] + a*stdDev_V[0];
	//float beta2 = mean_V[0] - a*stdDev_V[0];
	//float alpha1_H = mean_H[0] + a*stdDev_H[0];
	//float alpha2_H = mean_H[0] - a*stdDev_H[0];
	//float alpha1_S = mean_S[0] + a*stdDev_S[0];
	//float alpha2_S = mean_S[0] - a*stdDev_S[0];

	////std::cout << "beta1: " << beta1 << ", beta2: " << beta2 << std::endl;

	//for (int i = 0; i < frame.rows; i++)
	//{
	//	for (int j = 0; j < frame.cols; j++)
	//	{
	//	//	if (shadowMask.at<uint8_t>(i, j) == 0)
	//	//		continue;

	//		float Rp = R.at<float>(i, j);
	//		//std::cout << "Rp: " << Rp << std::endl;
	//		float Dp_H = D_H.at<float>(i, j);
	//		float Dp_S = D_S.at<float>(i, j);

	//		if (Rp > beta2 && Rp < beta1)
	//		{
	//			if (Dp_H > alpha2_H && Dp_H < alpha1_H)
	//			{
	//				if (Dp_S > alpha2_S && Dp_S < alpha1_S)
	//				{
	//					shadowMask2.at<uint8_t>(i, j) = 1;
	//				}
	//			}
	//		}
	//	}
	//}

	// merge shadow masks
	//bitwise_and(shadowMask, shadowMask2, shadowMask);

	// MRF
	const int num_labels = 3;
	GCoptimizationGridGraph gc(frameSILTP.cols, frameSILTP.rows, num_labels);
	gc.setSmoothCostFunctor(&this->fct);

	//Set the pixel-label probability
	for (int r = 2; r < frame.rows - 2; r++)
	{
		for (int c = 2; c < frame.cols - 2; c++)
		{
			int idx = frameSILTP.size().width*(r-2) + (c-2);

			if (gmm->BackgroundProbability.at<float>(r, c) == 0)
			{
				// we're pretty sure this pixel belongs to the background
				gc.setDataCost(idx, 0, -1.*log(0.95));
				gc.setDataCost(idx, 1, -1.*log(0.1));
				gc.setDataCost(idx, 2, -1.*log(0.2));
			}
			else
			{
				if(shadowMask.at<uint8_t>(r - 2, c - 2) == 1)
				{
					// shadow
					gc.setDataCost(idx, 0, -1.*log(0.1));
					gc.setDataCost(idx, 1, -1.*log(0.7));
					gc.setDataCost(idx, 2, -1.*log(0.2));
				}
				else
				{
					// foreground
					gc.setDataCost(idx, 0, -1.*log(0.1));
					gc.setDataCost(idx, 1, -1.*log(0.3));
					gc.setDataCost(idx, 2, -1.*log(0.7));
				}
			}
		}
	}

	//printf("Before optimization energy is %lld\n",gc.compute_energy());
	gc.expansion();
	//printf("After optimization energy is %lld\n",gc.compute_energy());

	for (int idx = 0; idx < frameSILTP.cols * frameSILTP.rows; idx++)
		shadowMask.at<uint8_t>(idx) = gc.whatLabel(idx) * (255/2);

	shadowMask.copyTo(_dst);
//	imshow("shadowMask", shadowMask * 255);
	//imshow("prob", gmm->BackgroundProbability);
	//imshow("fg texture", frameSILTP);
	//imshow("shadowMask2", shadowMask2 * 255);
}
