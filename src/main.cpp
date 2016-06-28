#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include "benedek.h"
#include "classifier.h"

using namespace cv;

const float scaleFactor = .5f;

#define MORPH_SIZE 0
#define MEDIAN_SIZE 3 
#define BENCHMARK_FRAMES_NUM 400

const std::string videoPath = "/mnt/things/car detection/videos/act.mp4";
BenedekSziranyi* benek = new BenedekSziranyi();
Classifier* classifier = new Classifier();

int main(int argc, char** argv)
{
	Mat inputFrame, foregroundMask, shadowMask, displayFrame, bgModel, morphKernel;
	VideoWriter writer;
	VideoCapture cap(videoPath);

	if (!cap.isOpened())
	{
		std::cout << "Could not open video file" << std::endl;
		return -1;
	}

	//cap.set(CAP_PROP_POS_MSEC, 1292000); // skip to 21:30
	//cap.set(CAP_PROP_POS_MSEC, (41*60+2)*1000); // skip to 41:42

	cap >> inputFrame;
	resize(inputFrame, inputFrame, Size(), scaleFactor, scaleFactor);
	benek->Init(inputFrame.size());
	
	bool benchmarkMode = false;
	bool record = false;
	bool loadNew = true;

	if(argc > 1 && std::string(argv[1]) == "-b")
	{
		std::cout << "Benchmark mode." << std::endl;
		benchmarkMode = true;
	}
	
	if(argc > 1 && std::string(argv[1]) == "-r")
	{
		std::cout << "Recording." << std::endl;
 		record = true;

		writer.open("tom.avi", VideoWriter::fourcc('X', 'V','I','D'),
				cap.get(CV_CAP_PROP_FPS), Size(inputFrame.size().width, 2*inputFrame.size().height));
	}

	if (!benchmarkMode)
	{
		namedWindow("OpenCV", WINDOW_AUTOSIZE);
	}

	if (MORPH_SIZE != 0)
		morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(MORPH_SIZE, MORPH_SIZE));

	uint32_t inputFrameNum = 0;
	auto t1 = std::chrono::steady_clock::now();

	while(true)
	{
		if(loadNew)
		{
			cap >> inputFrame;
			resize(inputFrame, inputFrame, Size(), scaleFactor, scaleFactor);
			
			benek->ProcessFrame(inputFrame, foregroundMask, shadowMask);

			if (MEDIAN_SIZE != 0)
				medianBlur(foregroundMask, foregroundMask, MEDIAN_SIZE);
			if (MORPH_SIZE != 0)
				morphologyEx(foregroundMask, foregroundMask, MORPH_OPEN, morphKernel);

			inputFrameNum++;

			if (!benchmarkMode)
			{
				Mat staufferForegroundMask, row1, row2;

				classifier->DrawBoundingBoxes(inputFrame, foregroundMask);
				displayFrame = inputFrame;
				cvtColor(displayFrame, displayFrame, COLOR_Luv2BGR);

				foregroundMask *= 255;
				//foregroundMask += 127 * shadowMask;
				cvtColor(foregroundMask, foregroundMask, COLOR_GRAY2BGR);

				staufferForegroundMask = benek->GetStaufferForegroundMask() * 255;
				cvtColor(staufferForegroundMask, staufferForegroundMask, COLOR_GRAY2BGR);

				hconcat(inputFrame, foregroundMask, row1);

				bgModel = benek->GetStaufferBackgroundModel();
				cvtColor(bgModel, bgModel, COLOR_Luv2BGR);
				hconcat(bgModel, staufferForegroundMask, row2);

				vconcat(row1, row2, displayFrame);
				imshow("OpenCV", displayFrame);
				
				if(record)
					writer << displayFrame;
			}
		}
		
		if (benchmarkMode && inputFrameNum == BENCHMARK_FRAMES_NUM)
			break;

		char key = waitKey(20);
		if(key == 'q')
			break;
		else if (key == ' ')
			loadNew = !loadNew;
		else if (key == 's')
			benek->ToggleShadowDetection();
	}

	auto t2 = std::chrono::steady_clock::now();
	if (benchmarkMode)
	{
		auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
		std::cout << "Processed " << BENCHMARK_FRAMES_NUM << " inputFrames in " << time_span.count() << " seconds." << std::endl;
	}

	if (record)
		writer.release();

	delete benek;
	delete classifier;
	return 0;
}

