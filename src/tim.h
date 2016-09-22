#ifndef TIM_H
#define TIM_H

#include <opencv2/videoio.hpp>
#include <nanomsg/nn.h>
#include <string>
#include "background.h"
#include "classifier.h"
#include "shadows.h"

#define BENCHMARK_FRAMES_NUM 400

using namespace std;

struct TimParameters
{
    std::string fileName;
    bool benchmark;
    bool record;
    bool classifyColours;
    bool removeShadows = false;
};

class Tim
{
    public:
        ~Tim();
        bool open(const TimParameters& params);
        void processFrames();

    private:
        TimParameters params;
        const double scaleFactor = .5;

        bool paused = false;
        uint32_t frameCount = 0;

        Background* background = nullptr;
        Shadows* shadows = nullptr;
        Classifier* classifier = nullptr;
        VideoCapture videoCapture;
        VideoWriter videoWriter;
        Size frameSize;
        Mat roiMask, objectLabels, objectLabelsCopy;

        // a copy is needed when playback is paused, but we want to update shadow detection params
        std::vector<MovingObject> movingObjects, movingObjectsCopy;

        int socket;

        void detectMovingObjects(InputArray _fgMask);
};

#endif

/* vim: set ft=cpp ts=4 sw=4 sts=4 tw=0 fenc=utf-8 et: */
