#include "shadows.h"
#include "background.h"
#include "siltp.h"
#include "gco/GCoptimization.h"

GCoptimization::EnergyTermType Shadows::MySmoothCostFunctor::compute(GCoptimization::SiteID s1, GCoptimization::SiteID s2, 
		GCoptimization::LabelID l1, GCoptimization::LabelID l2)
{
	if ((l1 - l2) * (l1 - l2) <= 2) 
		return (l1 - l2) * (l1 - l2);
	else 
		return 2;
}

Shadows::Shadows() : distanceThreshold(3), absoluteThreshold(7)
{
}

void Shadows::removeShadows(InputArray _src, InputArray _bg, InputArray _bgTexture, InputArray _fgMask, OutputArray _dst)
{
	Mat frame = _src.getMat(), background = _bg.getMat(), backgroundTexture = _bgTexture.getMat(), 
		foregroundMask = _fgMask.getMat(), shadowMask = _dst.getMat(), frameTexture;

	SILTP_16x2(frame, frameTexture);

	//cv::FileStorage fs("fgTexture.yml", cv::FileStorage::WRITE);
	//fs << "fgTexture" << frameTexture;

	for (int row = 2; row < frame.rows - 2; ++row)
	{
		uint32_t *bgTexturePtr = backgroundTexture.ptr<uint32_t>(row - 2);
		uint32_t *fgTexturePtr = frameTexture.ptr<uint32_t>(row - 2);
		uint8_t *shadowMaskPtr = shadowMask.ptr<uint8_t>(row - 2);
		uint8_t *bgLPtr = background.ptr<uint8_t>(row);
		uint8_t *fgLPtr = frame.ptr<uint8_t>(row);

		for (int col = 2; col < frame.cols - 2; col++)
		{
			uint32_t bgTexture = *bgTexturePtr++;
			uint32_t fgTexture = *fgTexturePtr++;

			uint8_t distance = hamming_distance(bgTexture, fgTexture);
			uint8_t bg = *bgLPtr; bgLPtr += 3;
			uint8_t fg = *fgLPtr; fgLPtr += 3; 

			*shadowMaskPtr++ = 
				((distance < distanceThreshold || fgTexture < absoluteThreshold) && fg/float(bg) <= 0.9);
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
	GCoptimizationGridGraph gc(frameTexture.cols, frameTexture.rows, num_labels);
	gc.setSmoothCostFunctor(&smoothFunctor);

	//Set the pixel-label probability
	for (int r = 2; r < frame.rows - 2; r++)
	{
		for (int c = 2; c < frame.cols - 2; c++)
		{
			int idx = frameTexture.size().width*(r-2) + (c-2);

			if (foregroundMask.at<uint8_t>(r, c) == 0)
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

	for (int row = 0; row < shadowMask.rows; ++row)
	{
		uint8_t *shadowMaskPtr = shadowMask.ptr<uint8_t>(row);
		int idx = shadowMask.cols * row; 

		for (int col = 0; col < shadowMask.cols; col++, idx++)
			*shadowMaskPtr++ = gc.whatLabel(idx) * (255/2);
	}

//	imshow("shadowMask", shadowMask * 255);
	//imshow("prob", gmm->BackgroundProbability);
	//imshow("fg texture", frameTexture);
	//imshow("shadowMask2", shadowMask2 * 255);
}
