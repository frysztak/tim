#include "shadows.h"
#include "background.h"
#include "siltp.h"
#include "GridCut/AlphaExpansion_2D_4C.h"

Shadows::Shadows(const Size& size) : distanceThreshold(7), absoluteThreshold(8), stdDevCoeff(0.2), numLabels(3)
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
		foregroundMask = _fgMask.getMat(), labelMask = _dst.getMat(), frameTexture, 
		shadowMaskHSV = Mat::zeros(frame.size(), CV_8U), 
		shadowMask = Mat::zeros(Size(frame.cols-4,frame.rows-4), CV_8U);

	SILTP_16x2(frame, frameTexture);

	Mat hammDistance = Mat::zeros(frameTexture.size(), CV_8U);
	Mat shadowMaskSILTP = Mat::zeros(frameTexture.size(), CV_8U);

	// detect shadows in colour space
	Mat bgHSV = Mat::zeros(frame.size(), CV_32F), fgHSV = Mat::zeros(frame.size(), CV_32F);
	cvtColor(frame, fgHSV, COLOR_BGR2HSV);
	cvtColor(background, bgHSV, COLOR_BGR2HSV);
	
	std::vector<Mat> fg_channels, bg_channels;
	split(fgHSV, fg_channels);
	split(bgHSV, bg_channels);

	Mat R, D_H, D_S;
	R = Mat::zeros(frame.size(), CV_32F);
	D_H = Mat::zeros(frame.size(), CV_32F);
	D_S = Mat::zeros(frame.size(), CV_32F);
	subtract(fg_channels[0], bg_channels[0], D_H, noArray(), CV_32F); // H
	subtract(fg_channels[1], bg_channels[1], D_S, noArray(), CV_32F); // S
	divide(fg_channels[2], bg_channels[2], R, 1, CV_32F); // V

	Scalar mean_H, stdDev_H;
	Scalar mean_S, stdDev_S;
	Scalar mean_V, stdDev_V;
	meanStdDev(D_H, mean_H, stdDev_H, foregroundMask);
	meanStdDev(D_S, mean_S, stdDev_S, foregroundMask);
	meanStdDev(R, mean_V, stdDev_V, foregroundMask);

	//float beta1 = mean_V[0] + a*stdDev_V[0];
	float beta2 = mean_V[0] - stdDevCoeff*stdDev_V[0];
	float alpha1_H = mean_H[0] + stdDevCoeff*stdDev_H[0];
	float alpha2_H = mean_H[0] - stdDevCoeff*stdDev_H[0];
	float alpha1_S = mean_S[0] + stdDevCoeff*stdDev_S[0];
	float alpha2_S = mean_S[0] - stdDevCoeff*stdDev_S[0];

	//std::cout << "beta1: " << beta1 << ", beta2: " << beta2 << std::endl;
	//std::cout << "mean_H: " << mean_H[0] << ", alpha1_H: " << alpha1_H << ", alpha2_H: " << alpha2_H << std::endl;
	//std::cout << "mean_S: " << mean_S[0] << ", alpha1_S: " << alpha1_S << ", alpha2_S: " << alpha2_S << std::endl;

	for (int i = 0; i < frame.rows; i++)
	{
		for (int j = 0; j < frame.cols; j++)
		{
			if (foregroundMask.at<uint8_t>(i, j) == 0)
				continue;

			float Rp = R.at<float>(i, j);
			float Dp_H = D_H.at<float>(i, j);
			float Dp_S = D_S.at<float>(i, j);

			if (Rp < beta2 && (Dp_H >= alpha1_H || Dp_H <= alpha2_H) && (Dp_S <= alpha2_S || Dp_S >= alpha1_S))
				shadowMaskHSV.at<uint8_t>(i, j) = 1;
		}
	}

	imshow("shadowMaskHSV", 255*shadowMaskHSV);

	// in texture space
	for (int row = 2; row < frame.rows - 2; ++row)
	{
		uint16_t *bgTexturePtr = backgroundTexture.ptr<uint16_t>(row - 2);
		uint16_t *fgTexturePtr = frameTexture.ptr<uint16_t>(row - 2);
		uint8_t *shadowMaskSILTPPtr = shadowMaskSILTP.ptr<uint8_t>(row - 2);
		uint8_t *hammDistancePtr = hammDistance.ptr<uint8_t>(row - 2);

		for (int col = 2; col < frame.cols - 2; col++)
		{
			uint16_t bgTexture = *bgTexturePtr++;
			uint16_t fgTexture = *fgTexturePtr++;

			uint8_t distance = hamming_distance(bgTexture, fgTexture);

			*hammDistancePtr++ = distance;
			*shadowMaskSILTPPtr++ = 
				((distance < distanceThreshold || fgTexture < absoluteThreshold));
		}
	}

	imshow("shadowMaskSILTP", 255*shadowMaskSILTP);
	//imshow("hamm distance", 15*hammDistance);
	
	// merge shadow masks
	bitwise_and(shadowMaskSILTP, shadowMaskHSV(Rect(2,2,shadowMaskHSV.cols-4,shadowMaskHSV.rows-4)), shadowMask);
	imshow("shadowMask", shadowMask * 255);

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
				dataCosts[(x+y*width)*numLabels+1] = hammDistance.at<uint8_t>(y, x);  //-10*log(0.7);
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
			int cost = 0;
			if (label != otherLabel && hammDistance.at<uint8_t>(y, x) > 50)
				cost = hammDistance.at<uint8_t>(y,x);
			smoothnessCosts[xy*2+0][label+otherLabel*numLabels] = cost; 
			smoothnessCosts[xy*2+1][label+otherLabel*numLabels] = cost; 
  		}
	}

	typedef AlphaExpansion_2D_4C<int,int,int> Expansion; 
	Expansion* expansion = new Expansion(width,height,numLabels,dataCosts,smoothnessCosts);
	expansion->perform();
	
	int* labeling = expansion->get_labeling();

	for (int row = 0; row < shadowMask.rows; ++row)
	{
		uint8_t *labelMaskPtr = labelMask.ptr<uint8_t>(row);
		int idx = labelMask.cols * row; 

		for (int col = 0; col < labelMask.cols; col++, idx++)
			*labelMaskPtr++ = labeling[idx] == 2 ? 255 : 0;
	}

	medianBlur(labelMask, labelMask, 3);

	//delete expansion;
}
