#ifndef MOVINGOBJECT_H
#define MOVINGOBJECT_H

#include <opencv2/opencv.hpp>

using namespace cv;

struct Segment
{
	Segment(Mat& mask, int area);

	Mat mask;
	int area;
};

class MovingObject
{
	private:
		int maxNumberOfFeatures;
		float featureQualityLevel;
		int minDistanceBetweenFeatures;

	public:
		MovingObject(const Size& size);

		std::vector<Segment> segments;
		Mat segmentLabels, mask, miniMask;
		Rect selector;

		// for tracking
		std::vector<Point2f> prevFeatures, features;
		uint32_t ID = 0, featuresLastUpdated = 0;
		bool remove = false;

		void minimizeMask();
		void updateTrackedFeatures(InputArray _grayFrame, uint32_t frameNumber);
};

#endif
