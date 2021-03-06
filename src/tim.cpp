#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <chrono>
#include <nanomsg/pair.h>
#include "tim.h"
#include "json11.hpp"

using namespace json11;

Tim::~Tim()
{
    if (background) delete background;
    if (shadows) delete shadows;
    if (classifier) delete classifier;

    nn_close(socket);
    std::remove("/tmp/tim.path");
}

bool Tim::open(const TimParameters& parameters)
{
#ifndef DATA_DIR
#error data dir is not defined.
#endif

    this->params = parameters;
    // open .json file and parse it
    string jsonFileName = DATA_DIR + params.fileName + ".json"; 
    ifstream jsonFile(jsonFileName, ifstream::in);
    Json json;

    if (jsonFile.is_open()) 
    {
        string err, jsonString((std::istreambuf_iterator<char>(jsonFile)), std::istreambuf_iterator<char>());
        json = Json::parse(jsonString, err);
    }
    else
    {
        std::cout << "couldn't open " << jsonFileName << std::endl;
        return false;
    }

    params.removeShadows = json["shadowDetection"].bool_value();
    double startTime = json["startTime"].number_value();
    std::string naturalDirection = json["naturalDirection"].string_value();
    
    // open video file
    string videoFileName = DATA_DIR + params.fileName + ".mp4"; 
    videoCapture.open(videoFileName);
    if (!videoCapture.isOpened())
    {
        cout << "could not open video file" << endl;
        return false;
    }

    // create temporary file containing full path to .json,
    // so that I don't have to specify anything when running Python scripts
    std::remove("/tmp/tim.path");
    std::ofstream file("/tmp/tim.path");
    file << jsonFileName;
    file.close();

    // create nanomsg socket. it's used to communicate with Python scripts.
    socket = nn_socket(AF_SP, NN_PAIR);
    if (socket >= 0)
        nn_connect(socket, "ipc:///tmp/tim.ipc");
    
    auto width = videoCapture.get(CV_CAP_PROP_FRAME_WIDTH);
    auto height = videoCapture.get(CV_CAP_PROP_FRAME_HEIGHT);
    auto fps = videoCapture.get(CV_CAP_PROP_FPS);
    videoCapture.set(CV_CAP_PROP_POS_MSEC, startTime * 1000);
    this->frameSize = Size(width * scaleFactor, height * scaleFactor);

    if (params.record)
    {
        videoWriter.open("demo.avi", VideoWriter::fourcc('X','V','I','D'), fps, Size(width, height));
        if (!videoWriter.isOpened())
        {
            cout << "could not open output video file" << endl;
            return false;
        }
    }

    // prepare ROI mask
    roiMask = Mat::zeros(frameSize, CV_8U);
    std::vector<Point> roiPoints, roiPolygon;
    for (const Json& list: json["roi"].array_items())
    {
        const Json& innerList = list.array_items();
        roiPoints.emplace_back(innerList[0].number_value()*frameSize.width, 
                               innerList[1].number_value()*frameSize.height);
    }
    approxPolyDP(roiPoints, roiPolygon, 1.0, true);
    fillConvexPoly(roiMask, &roiPolygon[0], roiPolygon.size(), 255, 8, 0); 

    // prepare collision lines
    std::vector<Point> linesPoints;
    for (const Json& list: json["lines"].array_items())
    {
        const Json& innerList = list.array_items();
        linesPoints.emplace_back(innerList[0].number_value()*frameSize.width, 
                                 innerList[1].number_value()*frameSize.height);
    }

    background = new Background(frameSize, json);
    shadows = new Shadows(json);
    classifier = new Classifier(linesPoints, naturalDirection);

    if (!params.benchmark)
        namedWindow("OpenCV", WINDOW_AUTOSIZE);
    else
        std::cout << "benchmark mode" << std::endl;

    return true;
}

void Tim::processFrames()
{
    Mat inputFrame, foregroundMask = Mat::zeros(frameSize, CV_8U), shadowMask, displayFrame, bgModel;

    auto t1 = std::chrono::high_resolution_clock::now();

    while (true)
    {
        if(!paused)
        {
            frameCount++;
            videoCapture >> inputFrame;
            resize(inputFrame, inputFrame, Size(), scaleFactor, scaleFactor);
            
#ifdef SIMD
            background->processFrameSIMD(inputFrame, foregroundMask);
#else
            background->processFrame(inputFrame, foregroundMask);
#endif
            foregroundMask &= roiMask;
            detectMovingObjects(foregroundMask);
        }

        if (paused)
        {
            movingObjects = movingObjectsCopy;
            objectLabels = objectLabelsCopy.clone();
        }

        shadowMask = Mat::zeros(frameSize, CV_8U);
        if (params.removeShadows)
        {
            shadows->removeShadows(inputFrame, background->getCurrentBackground(), 
                                   background->getCurrentStdDev(), foregroundMask, 
                                   objectLabels, movingObjects, shadowMask);
        }

        if (!params.benchmark)
        {
            Mat foregroundMaskBGR, row1, row2;

            inputFrame.copyTo(displayFrame);
            if (!paused)
            {
                Mat mask = params.removeShadows ? (shadowMask == 2) : foregroundMask;
                if (!params.dontTrack)
                {
                    classifier->trackObjects(displayFrame, mask, movingObjects);
                    classifier->checkCollisions();
                    classifier->updateCounters();
                    if (params.classifyColours)
                        classifier->classifyColours(displayFrame);
                }
            }

            if (!params.dontTrack)
            {
                classifier->drawBoundingBoxes(displayFrame, params.classifyColours);
                classifier->drawCollisionLines(displayFrame);
                classifier->drawCounters(displayFrame);
            }

            cvtColor(foregroundMask * 255, foregroundMaskBGR, COLOR_GRAY2BGR);
            hconcat(displayFrame, foregroundMaskBGR, row1);

            cvtColor(shadowMask * (255/2), shadowMask, COLOR_GRAY2BGR);

            hconcat(background->getCurrentBackground(), shadowMask, row2);
            vconcat(row1, row2, displayFrame);
            imshow("OpenCV", displayFrame);
            
            if(params.record)
                videoWriter << displayFrame;
        }
        
        if (params.benchmark && frameCount == BENCHMARK_FRAMES_NUM)
            break;

        if (!params.benchmark)
        {
            char key = waitKey(30);
            if(key == 'q')
                break;
            else if (key == ' ')
                paused = !paused;
            else if (key == 's')
                params.removeShadows = !params.removeShadows;

            // check if parameters got updated
            void *buf = NULL;
            int nbytes = nn_recv(socket, &buf, NN_MSG, NN_DONTWAIT);
            if (nbytes > 0)
            {
                std::string jsonString((const char*)buf, nbytes), err;
                auto json = Json::parse(jsonString, err);
                shadows->updateParameters(json);
                background->updateParameters(json);
                params.removeShadows = json["shadowDetection"].bool_value();
                nn_freemsg(buf);
            }
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    if (params.benchmark)
    {
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        std::cout << "processed " << BENCHMARK_FRAMES_NUM << " frames in " << time_span.count() 
                  << " seconds." << std::endl;
        std::cout << "average " << BENCHMARK_FRAMES_NUM / time_span.count() << " fps. " << std::endl;
    }
}

void Tim::detectMovingObjects(InputArray _fgMask)
{
    Mat fgMask = _fgMask.getMat();
    movingObjects.clear();

    // object masks: segment foreground mask into separate moving movingObjects
    int nLabels = connectedComponents(fgMask, objectLabels, 8, CV_16U);
    for (int label = 0; label < nLabels; label++)
        movingObjects.emplace_back(objectLabels.size());
    
    for (int idx = 0; idx < objectLabels.rows*objectLabels.cols; idx++)
    {
        uint16_t label = objectLabels.at<uint16_t>(idx);
        if (label == 0) continue;

        movingObjects[label].mask.at<uint8_t>(idx) = 1;
    }

    // remove tiniest objects 
    auto it = std::remove_if(movingObjects.begin(), movingObjects.end(), 
            [&](MovingObject& object) { return countNonZero(object.mask) < 100; });
    movingObjects.erase(it, movingObjects.end());

    for (auto& obj: movingObjects)
        obj.minimizeMask();

    movingObjectsCopy = movingObjects;
    objectLabelsCopy = objectLabels.clone();
}

/* vim: set ft=cpp ts=4 sw=4 sts=4 tw=0 fenc=utf-8 et: */
