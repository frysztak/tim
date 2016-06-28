#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <opencv2/opencv.hpp>

using namespace cv;

class Classifier
{
	public:
		void DrawBoundingBoxes(InputOutputArray _frame, InputArray _mask);
};

#endif
