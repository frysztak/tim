#ifndef COLOURCLASSIFIER_H
#define COLOURCLASSIFIER_H

#include <opencv2/core.hpp>
#include <tuple>
#include <vector>

using namespace cv;

class ColourClassifier
{
    private:
        std::vector<std::tuple<Scalar, std::string>> colourDictionary;
        Scalar BGR2Lab(float r, float g, float b);
        static float distance(const Scalar& a, const Scalar& b);

    public:
        ColourClassifier();
        std::string classifyColour(const Scalar& colour);
};

#endif
