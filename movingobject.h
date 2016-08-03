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
	Mat segmentLabels, mask;
	Rect selector;

	MovingObject(const Size& size)
	{
		mask = Mat::zeros(size, CV_8U);
	};
};

#endif
