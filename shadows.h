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

struct Segment
{
	Mat mask;
	int area;
};

struct Object
{
	std::vector<Segment> segments;
	Mat segmentLabels, mask;
	Rect selector;
};

class Shadows
{
	private:
		ShadowsParameters params;
				
		void findSegment(Object& object, Point startPoint, InputOutputArray _segmentLabels, 
				uint16_t label, float gradientThreshold);
		void fillInBlanks(InputArray _fgMask, InputArray _mask);
		void showSegmentation(int nSegments, InputArray _labels);
		void minimizeObjectMask(Object& obj);

		Mat objectLabels, D;
		
	public:
		Shadows(ShadowsParameters& params);
		void removeShadows(InputArray _src, InputArray _bg, InputArray _bgStdDev, InputArray _fgMask, OutputArray _dst);
};



#endif
