#include <vector>
#include <opencv2/opencv.hpp>
#include "stauffergrimson.h"

using namespace cv;

#define FOREGROUND_SECOND_PASS true

class BenedekSziranyi
{
	private:
		const float ForegroundThreshold = 10;
		const float ForegroundThreshold2 = 0.7;
		const uint8_t WindowSize = 10;
		const float Tau = 20;
		const float Kappa_min = 0.1;

		struct Shadow
		{
			float muL, muU, muV;
			float sigmaL, sigmaU, sigmaV;
		};

		struct Pixel
		{
			float L;
			float u;
			float v;
			float T;
			
			Shadow shadow;
		};
		
		std::vector<Pixel> Models;
		StaufferGrimson bgs;

		Mat ForegroundMask;
		void DetectForeground(InputArray _src);

	public:
		void Init(const Size& size);

		// output three-colour mask:
		// black: background
		// white: foreground
		// grey: shadows
		void ProcessFrame(InputArray _src, OutputArray _dst);
		const Mat& GetStaufferBackgroundModel();
		const Mat& GetStaufferForegroundMask();
};
