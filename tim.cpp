#include "tim.h"
#include "json11.hpp"
#include <fstream>
#include <iostream>
#include <chrono>

using namespace json11;

bool Tim::open(const string& name, bool benchmark, bool record)
{
	string fileName = "data/" + name + ".json";
	ifstream jsonFile(fileName, ifstream::in);
	if (!jsonFile.is_open()) 
	{
		cout << "can't open file: " << fileName << endl;
		return false;
	}

	string err;
	string jsonString((std::istreambuf_iterator<char>(jsonFile)), std::istreambuf_iterator<char>());

	auto json = Json::parse(jsonString, err);
	auto videoFile = dataRootDir + "videos/" + json["video"].string_value();
	benedek.foregroundThreshold = json["foregroundThreshold"].number_value();
	benedek.shadowDetectionEnabled = json["shadowDetectionEnabled"].bool_value();
	benedek.shadowModelUpdateRate = json["shadowUpdateRate"].int_value();
	benedek.Qmin = json["Qmin"].int_value();
	benedek.Qmax = json["Qmax"].int_value();
	benedek.tau = json["tau"].number_value();
	benedek.kappa_min = json["kappa_min"].number_value();

	benedek.windowPassEnabled = json["windowPassEnabled"].bool_value();
	benedek.windowSize = json["windowSize"].int_value();
	benedek.foregroundThreshold2 = json["foregroundThreshold2"].number_value();

	medianFilterSize = json["medianFilterSize"].int_value();
	morphFilterSize = json["morphKernel"].int_value();
	
	videoCapture.open(videoFile);
	if (!videoCapture.isOpened())
	{
		cout << "could not open video file" << endl;
		return false;
	}

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
	benedek.Init(this->frameSize);
	shadows.Init(&benedek.bgs);

	if (morphFilterSize != 0)
		morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(morphFilterSize, morphFilterSize));

	if (!benchmark)
		namedWindow("OpenCV", WINDOW_AUTOSIZE);
	else
		std::cout << "benchmark mode" << std::endl;

	this->benchmarkMode = benchmark;
	this->record = record;

	return true;
}

void Tim::processFrames()
{
	Mat inputFrame, foregroundMask, shadowMask, displayFrame, bgModel;

	auto t1 = std::chrono::steady_clock::now();

	while (true)
	{
		if(!paused)
		{
			videoCapture >> inputFrame;
			resize(inputFrame, inputFrame, Size(), scaleFactor, scaleFactor);
			
			benedek.ProcessFrame(inputFrame, foregroundMask, shadowMask);
			shadows.RemoveShadows(inputFrame, benedek.GetStaufferBackgroundModel(), shadowMask);

			if (medianFilterSize != 0)
				medianBlur(foregroundMask, foregroundMask, medianFilterSize);
			//if (morphFilterSize != 0)
			//	morphologyEx(foregroundMask, foregroundMask, MORPH_OPEN, morphKernel);

			frameCount++;

			if (!benchmarkMode)
			{
				Mat staufferForegroundMask, row1, row2, noShadow;

				//classifier.DrawBoundingBoxes(inputFrame, foregroundMask);
				displayFrame = inputFrame;
				cvtColor(displayFrame, displayFrame, COLOR_Luv2BGR);
				cvtColor(foregroundMask * 255, foregroundMask, COLOR_GRAY2BGR);

				hconcat(inputFrame, foregroundMask, row1);

				bgModel = benedek.GetStaufferBackgroundModel();
				cvtColor(bgModel, bgModel, COLOR_Luv2BGR);

				Mat cols = Mat::zeros(shadowMask.rows, 2, CV_8U);
				hconcat(cols, shadowMask, shadowMask);
				hconcat(shadowMask, cols, shadowMask);
				Mat rows = Mat::zeros(2, shadowMask.cols, CV_8U);
				vconcat(rows, shadowMask, shadowMask);
				vconcat(shadowMask, rows, shadowMask);
				cvtColor(shadowMask, shadowMask, COLOR_GRAY2BGR);

				hconcat(bgModel, shadowMask, row2);
				vconcat(row1, row2, displayFrame);
				imshow("OpenCV", displayFrame);
				
				if(record)
					videoWriter << displayFrame;
			}
		}
		
		if (benchmarkMode && frameCount == BENCHMARK_FRAMES_NUM)
			break;

		char key = waitKey(30);
		if(key == 'q')
			break;
		else if (key == ' ')
			paused = !paused;
		else if (key == 's')
			benedek.ToggleShadowDetection();

	}

	auto t2 = std::chrono::steady_clock::now();
	if (benchmarkMode)
	{
		auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
		std::cout << "processed " << BENCHMARK_FRAMES_NUM << " frames in " << time_span.count() << " seconds." << std::endl;
		std::cout << "average " << BENCHMARK_FRAMES_NUM / time_span.count() << " fps. " << std::endl;
	}
}
