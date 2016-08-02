#ifndef SHADOWS_H
#define SHADOWS_H

#include <opencv2/opencv.hpp>

using namespace cv;

struct ShadowsParameters
{
	float gradientThreshold, gradientThresholdMultiplier, lambda, tau, alpha, luminanceThreshold;
	bool edgeCorrection, autoGradientThreshold;
	int minSegmentSize;
};

class Shadows
{
	private:
		ShadowsParameters params;
				
		int findGSCN(Point startPoint, InputArray _objectMask, InputArray _D, InputOutputArray _labels, 
				InputOutputArray _binaryMask, uint16_t label, float gradientThreshold);
		void fillInBlanks(InputArray _fgMask, InputArray _mask);
		void showSegmentation(int nSegments, InputArray _labels);
		
	public:
		Shadows(ShadowsParameters& params);
		void removeShadows(InputArray _src, InputArray _bg, InputArray _bgStdDev, InputArray _fgMask, OutputArray _dst);
};

struct Segment
{
	Mat mask;
	int area;
};

struct Object
{
	std::vector<Segment> segments;
	Mat segmentLabels, mask;
};


#endif
