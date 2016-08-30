#ifndef MOVINGOBJECT_H
#define MOVINGOBJECT_H

#include <opencv2/opencv.hpp>

using namespace cv;

struct Segment
{
	Segment(Mat& mask, int area);
	Segment(const Segment& other);

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
		MovingObject() = default;
		MovingObject(const Size& size);
		MovingObject(const MovingObject& other);
		MovingObject& operator=(const MovingObject& other);

		std::vector<Segment> segments;
		Mat segmentLabels, mask, miniMask;
		Rect selector;

		// for tracking
		std::vector<Point2f> prevFeatures, features;
		uint32_t ID = 0, featuresLastUpdated = 0;
		bool remove = false;

		void minimizeMask();
		void updateTrackedFeatures(InputArray _grayFrame, uint32_t frameNumber);
		void predictNextPosition(InputArray _prevGrayFrame, InputArray _grayFrame);
};

#endif
