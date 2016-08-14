#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

using namespace cv;

class Classifier
{
	public:
		void DrawBoundingBoxes(InputOutputArray _frame, InputArray _mask);

	private:
		struct Object
		{
			Rect2d rect;
			Ptr<Tracker> tracker;
			int ID = 0;

			bool doRectsOverlap(const Rect2d& newRect) const
			{
				auto r = this->rect & newRect;
				if (r.area() > 0)
					return true;

				return false;
			}
		};

		int objCounter = 0;
		std::vector<Object> objects;
};

#endif
