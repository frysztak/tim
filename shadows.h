#ifndef SHADOWS_H
#define SHADOWS_H

#include "movingobject.h"

using namespace cv;

struct ShadowsParameters
{
	float gradientThreshold, gradientThresholdMultiplier, lambda, tau, alpha, luminanceThreshold;
	bool edgeCorrection, autoGradientThreshold;
	int minObjectSize, minSegmentSize;

	void parse(const std::string& json);
};

class Shadows
{
	private:
		ShadowsParameters params;
				
		void findSegment(MovingObject& object, Point startPoint, InputOutputArray _segmentLabels, 
				uint16_t label, float gradientThreshold);
		void fillInBlanks(InputArray _fgMask, InputArray _mask);
		void minimizeObjectMask(MovingObject& obj);
		void showSegmentation(int nSegments, InputArray _labels);

		Mat objectLabels, D;
		
	public:
		Shadows(const std::string& jsonString);
		void updateParameters(const std::string& jsonString);
		void removeShadows(InputArray _src, InputArray _bg, InputArray _bgStdDev, InputArray _fgMask, OutputArray _dst);
};



#endif
