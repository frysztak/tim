#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include "movingobject.h"

using namespace cv;

class Classifier
{
	public:
		void trackObjects(InputArray _frame, InputArray _fgMask, std::vector<MovingObject>& objects);
		void drawBoundingBoxes(InputOutputArray _frame);

	private:
		Mat prevFrame;
		int frameCounter = 0;
		int objCounter = 0;
		std::vector<MovingObject> classifiedObjects;
};

#endif
