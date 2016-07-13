#include "shadows.h"
#include "background.h"
#include "siltp.h"
#include "GridCut/AlphaExpansion_2D_4C.h"

Shadows::Shadows(const Size& size) : distanceThreshold(3), absoluteThreshold(7), numLabels(3)
{
	dataCosts = new int[size.area()*numLabels];

	smoothnessCosts = new int*[size.area()*2];

	for(int y = 0; y < size.height; y++)
	for(int x = 0; x < size.width; x++)
	{    
		const int xy = x+y*size.width;
		
		smoothnessCosts[xy*2+0] = new int[numLabels*numLabels];
		smoothnessCosts[xy*2+1] = new int[numLabels*numLabels];
	}
}

Shadows::~Shadows()
{
	delete[] dataCosts;
	delete[] smoothnessCosts;
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
	const int width = frameTexture.cols;
	const int height = frameTexture.rows;

	//Set the pixel-label probability
	for(int y = 0; y < height; y++)
	for(int x = 0; x < width; x++)
	{
		if (foregroundMask.at<uint8_t>(y + 2, x + 2) == 0)
		{
			// pretty sure this pixel belongs to the background
			dataCosts[(x+y*width)*numLabels+0] = -10*log(0.9);
			dataCosts[(x+y*width)*numLabels+1] = -10*log(0.1);
			dataCosts[(x+y*width)*numLabels+2] = -10*log(0.2);
		}
		else
		{
			if(shadowMask.at<uint8_t>(y, x) == 1)
			{
				// shadow
				dataCosts[(x+y*width)*numLabels+0] = -10*log(0.1);
				dataCosts[(x+y*width)*numLabels+1] = -10*log(0.7);
				dataCosts[(x+y*width)*numLabels+2] = -10*log(0.15);
			}
			else
			{
				// foreground
				dataCosts[(x+y*width)*numLabels+0] = -10*log(0.1);	
				dataCosts[(x+y*width)*numLabels+1] = -10*log(0.3);
				dataCosts[(x+y*width)*numLabels+2] = -10*log(0.7);
			}
		}
	}

	for(int y = 0; y < height; y++)
	for(int x = 0; x < width; x++)
	{    
	  const int xy = x+y*width;
	
	  for(int label=0;label<numLabels;label++)
	  for(int otherLabel=0;otherLabel<numLabels;otherLabel++)
	  {
	    #define WEIGHT(A) (1+1000*std::exp(-((A)*(A))/5))
	
	    smoothnessCosts[xy*2+0][label+otherLabel*numLabels] = (label!=otherLabel && frameTexture.at<uint32_t>(y,x) > 1000) ? 25 : 0;
	    smoothnessCosts[xy*2+1][label+otherLabel*numLabels] = (label!=otherLabel && frameTexture.at<uint32_t>(y,x) > 1000) ? 25 : 0;

	//	smoothnessCosts[xy*2+0][label+otherLabel*numLabels] = (x<width-1  && label!=otherLabel) ? WEIGHT(frameTexture.at<uint8_t>(y,x+1) - frameTexture.at<uint8_t>(y,x)) : 0;
	//	smoothnessCosts[xy*2+1][label+otherLabel*numLabels] = (y<height-1 && label!=otherLabel) ? WEIGHT(frameTexture.at<uint8_t>(y+1,x) - frameTexture.at<uint8_t>(y,x)) : 0;
	    #undef WEIGHT
	  }
	}

	typedef AlphaExpansion_2D_4C<int,int,int> Expansion; 
	Expansion* expansion = new Expansion(width,height,numLabels,dataCosts,smoothnessCosts);
	expansion->perform();
	
	int* labeling = expansion->get_labeling();

	for (int row = 0; row < shadowMask.rows; ++row)
	{
		uint8_t *shadowMaskPtr = shadowMask.ptr<uint8_t>(row);
		int idx = shadowMask.cols * row; 

		for (int col = 0; col < shadowMask.cols; col++, idx++)
			*shadowMaskPtr++ = labeling[idx] * (255/2);
	}

	//delete expansion;

//	imshow("shadowMask", shadowMask * 255);
	//imshow("prob", gmm->BackgroundProbability);
	//imshow("fg texture", frameTexture);
	//imshow("shadowMask2", shadowMask2 * 255);
}
