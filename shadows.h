#ifndef SHADOWS_H
#define SHADOWS_H

#include <opencv2/opencv.hpp>

using namespace cv;

class Shadows
{
	private:
		const uint8_t distanceThreshold;
		const uint8_t absoluteThreshold;
		const int numLabels;
		int** smoothnessCosts = nullptr;
		int* dataCosts = nullptr;

	public:
		Shadows(const Size& size);
		~Shadows();
		void removeShadows(InputArray _src, InputArray _bg, InputArray _bgTexture, InputArray _fgMask, OutputArray _dst);
};

#endif
