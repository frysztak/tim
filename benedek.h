#ifndef BENEDEK_H
#define BENEDEK_H

#include <vector>
#include <tuple>
#include <opencv2/opencv.hpp>
#include "stauffergrimson.h"

using namespace cv;

class BenedekSziranyi
{
	friend class Tim;

	private:
		float foregroundThreshold = 8.5;
		float foregroundThreshold2 = 0.7;

		bool windowPassEnabled = true;
		uint8_t windowSize = 10;
		float tau = 15;
		float kappa_min = 0.1;

		bool shadowDetectionEnabled = true;
		int shadowModelUpdateRate = 150; // frames
		uint Qmin = 15'000;
		uint Qmax = 30'000;

		struct Shadow
		{
			float L_mean, u_mean, v_mean;
			float L_variance, u_variance, v_variance;
			std::vector<float> Wu_t; // there's no need to Wv_t. its content would be exactly the same as Wu_t
			std::vector<std::tuple<float, uint32_t>> Q;
			uint32_t lastUpdate = 0; // in frames
		};

		struct Pixel
		{
			float L;
			float u;
			float v;
			float T;
		};
	
		uint32_t currentFrame;	
		StaufferGrimson bgs;
		Shadow shadowModel;

		Mat ForegroundMask, ShadowMask;
		Size FrameSize;
		void DetectForeground(InputArray _src);
		void UpdateShadowModel();

	public:
		void Init(const Size& size);
		void InitShadowModel();

		// output three-colour mask:
		// black: background
		// white: foreground
		// grey: shadows
		
		void ProcessFrame(InputArray _src, OutputArray _fg, OutputArray _sh);
		void ToggleShadowDetection();
		const Mat& GetStaufferBackgroundModel();
		const Mat& GetStaufferForegroundMask();
};

#endif
