#ifndef MOVINGOBJECT_H
#define MOVINGOBJECT_H

#include <opencv2/core.hpp>
#include <map>

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
        Scalar colour;

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
        bool alreadyCounted = false;

        std::map<int, uint32_t> collisions;

        void minimizeMask();
        void updateTrackedFeatures(InputArray _grayFrame, uint32_t frameNumber);
        void predictNextPosition(InputArray _prevGrayFrame, InputArray _grayFrame);

        std::string colourString;
        void averageColour(InputArray _frame);
        Scalar getColour();
};

#endif

/* vim: set ft=cpp ts=4 sw=4 sts=4 tw=0 fenc=utf-8 et: */
