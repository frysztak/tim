#include "colourclassifier.h"
#include <opencv2/imgproc.hpp>

Scalar ColourClassifier::BGR2Lab(float b, float g, float r)
{
    Mat mat(1, 1, CV_32FC3);
    mat.at<Vec3f>(0) = Vec3f(b, g, r) / 255;
    cvtColor(mat, mat, COLOR_BGR2Lab);

    return Scalar(mat.at<Vec3f>(0)[0],
                  mat.at<Vec3f>(0)[1],
                  mat.at<Vec3f>(0)[2]);
}

ColourClassifier::ColourClassifier()
{
    this->colourDictionary = 
    {
        std::make_tuple(BGR2Lab(255, 255, 255), "white"),
        std::make_tuple(BGR2Lab(0,   0,   0),   "black"),
        std::make_tuple(BGR2Lab(128, 128, 128), "gray"),
        std::make_tuple(BGR2Lab(97,  89,  84),  "silver"),
        std::make_tuple(BGR2Lab(84,  66,  103), "dark red"),
    };
}

float ColourClassifier::distance(const Scalar& a, const Scalar& b)
{
    float distance = 0; // CIE76
    distance += (a[0] - b[0]) * (a[0] - b[0]);
    distance += (a[1] - b[1]) * (a[1] - b[1]);
    distance += (a[2] - b[2]) * (a[2] - b[2]);
    return sqrt(distance);
}

std::string ColourClassifier::classifyColour(const Scalar& bgr)
{
    auto lab = BGR2Lab(bgr[0], bgr[1], bgr[2]);
    auto closestColour = *std::min_element(std::begin(colourDictionary), 
                                           std::end(colourDictionary),
                                           [&](auto& a, auto& b) 
                                           { 
                                               return distance(std::get<0>(a), lab) <
                                                      distance(std::get<0>(b), lab); 
                                           });
    return std::get<1>(closestColour);
}
