#ifndef SHADOWS_H
#define SHADOWS_H

#include <opencv2/opencv.hpp>

using namespace cv;

class Shadows
{
	private:
		// texture-related coeffients
		const int distanceThreshold;
		const int absoluteThreshold;
		// colour-related coeffient
		const float stdDevCoeff;
		// MRF
		const int numLabels;
		int** smoothnessCosts = nullptr;
		int* dataCosts = nullptr;

	public:
		Shadows(const Size& size);
		~Shadows();
		void removeShadows(InputArray _src, InputArray _bg, InputArray _bgTexture, InputArray _fgMask, OutputArray _dst);
};

#endif
