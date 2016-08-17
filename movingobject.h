#ifndef MOVINGOBJECT_H
#define MOVINGOBJECT_H

#include <opencv2/opencv.hpp>

using namespace cv;

struct Segment
{
	Mat mask;
	int area;

	Segment(Mat& mask, int area) : mask(mask), area(area) { };
};

struct MovingObject
{
	std::vector<Segment> segments;
	Mat segmentLabels, mask, miniMask;
	Rect selector;

	// for tracking
	std::vector<Point2f> prevFeatures, features;
	uint32_t ID = 0, featuresLastUpdated = 0;
	bool remove = false;

	MovingObject() {};
	MovingObject(const Size& size)
	{
		mask = Mat::zeros(size, CV_8U);
	};

	void minimizeMask()
	{
		Rect rect = boundingRect(this->mask);
		Size newSize = rect.size();
		auto offset = rect.tl();

		this->selector = Rect(offset.x, offset.y, newSize.width, newSize.height);
		this->miniMask = this->mask(this->selector);
	}
};

#endif
