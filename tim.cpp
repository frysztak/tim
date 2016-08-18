#include "tim.h"
#include "json11.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <nanomsg/pair.h>

using namespace json11;

bool Tim::open(const string& name, bool benchmark, bool record)
{
	string fileName = dataRootDir + "json/" + name + ".json";
	ifstream jsonFile(fileName, ifstream::in);
	if (!jsonFile.is_open()) 
	{
		cout << "can't open file: " << fileName << endl;
		return false;
	}

	string err, jsonString((std::istreambuf_iterator<char>(jsonFile)), std::istreambuf_iterator<char>());
	auto json = Json::parse(jsonString, err);

	medianFilterSize = json["medianFilterSize"].int_value();
	morphFilterSize = json["morphKernel"].int_value();
	
	auto videoFileName = dataRootDir + "videos/" + json["video"].string_value();
	videoCapture.open(videoFileName);
	if (!videoCapture.isOpened())
	{
		cout << "could not open video file" << endl;
		return false;
	}

	// nn_send blocks the thread, so start a separate one just for a while
	// nanomsg sockets are thread-safe, thank god
	std::thread sendJsonFilename_thread([&]()
	{
		socket = nn_socket(AF_SP, NN_PAIR);
		if (socket >= 0)
		{
			if (nn_bind(socket, "ipc:///tmp/tim.ipc") >= 0)
			{
				const char* msg = fileName.c_str();
				nn_send(socket, msg, strlen(msg), 0);
			}
		}
	});
	sendJsonFilename_thread.detach();
	
	auto width = videoCapture.get(CV_CAP_PROP_FRAME_WIDTH);
	auto height = videoCapture.get(CV_CAP_PROP_FRAME_HEIGHT);
	auto fps = videoCapture.get(CV_CAP_PROP_FPS);

	if (record)
	{
		videoWriter.open("demo.avi", VideoWriter::fourcc('X','V','I','D'), fps, Size(width, height));
		if (!videoWriter.isOpened())
		{
			cout << "could not open output video file" << endl;
			return false;
		}
	}

	this->frameSize = Size(width * scaleFactor, height * scaleFactor);
	background.init(this->frameSize);
	shadows = new Shadows(json);

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

	if (morphFilterSize != 0)
		morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(morphFilterSize, morphFilterSize));

	if (!benchmark)
		namedWindow("OpenCV", WINDOW_AUTOSIZE);
	else
		std::cout << "benchmark mode" << std::endl;

	this->benchmarkMode = benchmark;
	this->record = record;

	videoCapture.set(CV_CAP_PROP_POS_MSEC, 36*1000);

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
			background.processFrameSIMD(inputFrame, foregroundMask);
#else
			background.processFrame(inputFrame, foregroundMask);
#endif
			foregroundMask &= roiMask;
			detectMovingObjects(foregroundMask);
		}

		shadowMask = Mat::zeros(frameSize, CV_8U);
		if (removeShadows)
			shadows->removeShadows(inputFrame, background.getCurrentBackground(), background.getCurrentStdDev(), 
					foregroundMask, shadowMask);

		if (medianFilterSize != 0)
			medianBlur(foregroundMask, foregroundMask, medianFilterSize);
		//if (morphFilterSize != 0)
		//	morphologyEx(foregroundMask, foregroundMask, MORPH_OPEN, morphKernel);

		if (!benchmarkMode && !paused)
		{
			Mat foregroundMaskBGR, row1, row2;

			inputFrame.copyTo(displayFrame);
			//classifier.DrawBoundingBoxes(displayFrame, (shadowMask == 2) & roiMask, roiMask);	
			classifier.DrawBoundingBoxes(displayFrame, foregroundMask, movingObjects);
			cvtColor(foregroundMask * 255, foregroundMaskBGR, COLOR_GRAY2BGR);
			hconcat(displayFrame, foregroundMaskBGR, row1);

			cvtColor(shadowMask * (255/2), shadowMask, COLOR_GRAY2BGR);

			hconcat(background.getCurrentBackground(), shadowMask, row2);
			vconcat(row1, row2, displayFrame);
			imshow("OpenCV", displayFrame);
			
			if(record)
				videoWriter << displayFrame;
		}
		
		if (benchmarkMode && frameCount == BENCHMARK_FRAMES_NUM)
			break;

		if (!benchmarkMode)
		{
			char key = waitKey(30);
			if(key == 'q')
				break;
			else if (key == ' ')
				paused = !paused;
			else if (key == 's')
				removeShadows = !removeShadows;

			// check if parameters got updated
			void *buf = NULL;
			int nbytes = nn_recv(socket, &buf, NN_MSG, NN_DONTWAIT);
			if (nbytes > 0)
			{
				std::string jsonString((const char*)buf, nbytes), err;
				auto json = Json::parse(jsonString, err);
				shadows->updateParameters(json);
				nn_freemsg(buf);
			}
		}
	}

	auto t2 = std::chrono::high_resolution_clock::now();
	if (benchmarkMode)
	{
		auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
		std::cout << "processed " << BENCHMARK_FRAMES_NUM << " frames in " << time_span.count() << " seconds." << std::endl;
		std::cout << "average " << BENCHMARK_FRAMES_NUM / time_span.count() << " fps. " << std::endl;
	}
}

void Tim::detectMovingObjects(InputArray _fgMask)
{
	Mat fgMask = _fgMask.getMat(), objectLabels;
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
}
