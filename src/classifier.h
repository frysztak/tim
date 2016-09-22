#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <opencv2/core.hpp>
#include "movingobject.h"
#include "line.h"
#include "direction.h"
#include "colourclassifier.h"

using namespace cv;

class Classifier
{
    public:
        Classifier(const std::vector<Point>& collisionLines, const std::string& directionStr);

        void trackObjects(InputArray _frame, InputArray _fgMask, std::vector<MovingObject>& objects);
        void checkCollisions();
        void updateCounters();
        void classifyColours(InputArray _frame);

        void drawBoundingBoxes(InputOutputArray _frame, bool classifyColours);
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

        ColourClassifier colourClassifier;
};

#endif

/* vim: set ft=cpp ts=4 sw=4 sts=4 tw=0 fenc=utf-8 et: */
