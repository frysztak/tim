#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include "movingobject.h"
#include "line.h"
#include "direction.h"

using namespace cv;

class Classifier
{
	public:
		Classifier(const std::vector<Point>& collisionLines, const std::string& directionStr);

		void trackObjects(InputArray _frame, InputArray _fgMask, std::vector<MovingObject>& objects);
		void checkCollisions();
		void updateCounters();

		void drawBoundingBoxes(InputOutputArray _frame);
		void drawCollisionLines(InputOutputArray _frame);
		void drawCounters(InputOutputArray _frame);

	private:
		Mat prevFrame;
		int frameCounter = 0;
		int objCounter = 0;
		std::vector<MovingObject> classifiedObjects;

		Line collisionLines[2];
		// naturalDirection goes from line #0 to line #1
		Direction naturalDirection, oppositeDirection;
};

#endif
