#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include "movingobject.h"

using namespace cv;

class Classifier
{
	public:
		void DrawBoundingBoxes(InputOutputArray _frame, InputArray _fgMask, std::vector<MovingObject>& objects);

	private:
		struct Object
		{
			Rect2d rect;
			Mat mask;
			int featuresLastUpdated = 0;
			std::vector<Point2f> prevFeatures, features;
			int ID = 0;

			bool doRectsOverlap(Rect2d newRect) const
			{
				auto r = this->rect & (newRect + Size2d(10,10));
				if (r.area() > 0) return true;
				return false;
			}

			friend bool operator==(const Object& lhs, const Object& rhs)
			{
				return lhs.ID == rhs.ID; 
			}
		};

		Mat prevFrame;
		int frameCounter = 0;
		int objCounter = 0;
		std::vector<MovingObject> classifiedObjects;
};

#endif
