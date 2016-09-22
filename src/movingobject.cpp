#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#ifdef DEBUG
#include <iostream>
#endif
#include "movingobject.h"

Segment::Segment(Mat& mask, int area) : 
    mask(mask), area(area) 
{ 
}

Segment::Segment(const Segment& other) : 
    mask(other.mask.clone()), area(other.area)
{
}

MovingObject::MovingObject(const Size& size) :
    maxNumberOfFeatures(10),
    featureQualityLevel(0.01),
    minDistanceBetweenFeatures(8)
{
    mask = Mat::zeros(size, CV_8U);
}

MovingObject::MovingObject(const MovingObject& other) : 
   maxNumberOfFeatures(other.maxNumberOfFeatures), featureQualityLevel(other.featureQualityLevel),
   minDistanceBetweenFeatures(other.minDistanceBetweenFeatures), colour(other.colour),
   segments(other.segments), segmentLabels(other.segmentLabels), mask(other.mask.clone()),
   selector(other.selector),
   prevFeatures(other.prevFeatures), features(other.features), ID(other.ID), 
   featuresLastUpdated(other.featuresLastUpdated), remove(other.remove), alreadyCounted(other.alreadyCounted),
   collisions(other.collisions), colourString(other.colourString)
{
    miniMask = mask(selector);
}

MovingObject& MovingObject::operator=(const MovingObject& other)
{
    this->maxNumberOfFeatures = other.maxNumberOfFeatures;
    this->featureQualityLevel = other.featureQualityLevel;
    this->minDistanceBetweenFeatures = other.minDistanceBetweenFeatures;
    this->colour = other.colour;
    this->colourString = other.colourString;
    this->segments = std::vector<Segment>(other.segments);
    this->segmentLabels = other.segmentLabels;
    this->mask = other.mask.clone();
    this->selector = other.selector;
    this->prevFeatures = std::vector<Point2f>(prevFeatures);
    this->features = std::vector<Point2f>(features);
    this->featuresLastUpdated = other.featuresLastUpdated;
    this->remove = other.remove;
    this->alreadyCounted = other.alreadyCounted;
    this->collisions = other.collisions;
    this->miniMask = mask(selector);

    return *this;
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
    Mat frame = _grayFrame.getMat();
    
    std::vector<Point2f> newFeatures;
    goodFeaturesToTrack(frame(selector), 
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

void MovingObject::predictNextPosition(InputArray _prevGrayFrame, InputArray _grayFrame)
{
    Mat prevFrame = _prevGrayFrame.getMat(), frame = _grayFrame.getMat();

    std::vector<uint8_t> status;
    std::vector<float> err;
    features.clear();
    
    calcOpticalFlowPyrLK(prevFrame, 
                         frame,
                         prevFeatures,
                         features,
                         status,
                         err);

#ifdef DEBUG
    std::cout << "ID: " << ID << ", status: ";
    for (auto s: status)
        std::cout << (unsigned)s << " ";
    std::cout << ", err: ";
    for (float e: err)
        std::cout << e << ", ";
    std::cout << std::endl;
#endif

    // none of points matched, mark object for deletion
    if (std::all_of(err.begin(), err.end(), [](float e) { return e < 2; }))
        remove = true;

    // remove points that could not be tracked
    for (int i = 0; i < (int)err.size(); i++)
        if (err[i] < 2) features.erase(features.begin() + i);
}

void MovingObject::averageColour(InputArray _frame)
{
    Mat frame = _frame.getMat();
    auto c = mean(frame(selector), miniMask);
    if (colour[0] == 0 && colour[1] == 0 && colour[2] == 0)
        colour = c; 
    else
        colour = (colour + c) / 2;
}

Scalar MovingObject::getColour()
{
    return this->colour;
}

/* vim: set ft=cpp ts=4 sw=4 sts=4 tw=0 fenc=utf-8 et: */
