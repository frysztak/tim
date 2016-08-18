#include "movingobject.h"

Segment::Segment(Mat& mask, int area) : 
	mask(mask), area(area) 
{ 
}

MovingObject::MovingObject(const Size& size) :
	maxNumberOfFeatures(10),
	featureQualityLevel(0.01),
	minDistanceBetweenFeatures(8)
{
	mask = Mat::zeros(size, CV_8U);
}

void MovingObject::minimizeMask()
{
	Rect rect = boundingRect(mask);
	Size newSize = rect.size();
	auto offset = rect.tl();

	selector = Rect(offset.x, offset.y, newSize.width, newSize.height);
	miniMask = mask(selector);
}

void MovingObject::updateTrackedFeatures(InputArray _grayFrame, uint32_t frameNumber)
{
	Mat grayFrame = _grayFrame.getMat();
	
	std::vector<Point2f> newFeatures;
	goodFeaturesToTrack(grayFrame(selector), 
						newFeatures,	
						maxNumberOfFeatures,
						featureQualityLevel,
						minDistanceBetweenFeatures,
						miniMask);

	if (newFeatures.size() > 2)
	{
		for(auto& pt: newFeatures)
			pt += (Point2f)selector.tl();
	
		features = newFeatures;
		featuresLastUpdated = frameNumber;
	}
}
