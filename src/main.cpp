#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include "stauffergrimson.h"

using namespace cv;

const float scaleFactor = .5f;

#define MORPH_SIZE 2
#define MEDIAN_SIZE 3
#define BENCHMARK_FRAMES_NUM 400

const std::string videoPath = "/mnt/things/car detection/videos/auburn_toomers2.mp4";
StaufferGrimson* bgs = new StaufferGrimson();

void MouseEvent(int event, int x, int y, int flags, void* userdata)
{
	Mat* rgb = (Mat*)userdata;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		if (y > (rgb->rows / 2))
			y -= rgb->rows/2;
		
 		int idx = rgb->size().width * y + x; 
		printf("%d %d: %d, %d, %d\n", 
				x, y, 
				(int)(*rgb).at<Vec3b>(y, x)[2], 
				(int)(*rgb).at<Vec3b>(y, x)[1], 
				(int)(*rgb).at<Vec3b>(y, x)[0]);
		bgs->Dump(idx);
	}         
}

int main(int argc, char** argv)
{
	Mat inputFrame, foregroundMask, displayFrame, bgModel, morphKernel;
	VideoWriter writer;
	VideoCapture cap(videoPath);

	if (!cap.isOpened())
	{
		std::cout << "Could not open video file" << std::endl;
		return -1;
	}

	//cap.set(CAP_PROP_POS_MSEC, 1292000); // skip to 21:30
	cap.set(CAP_PROP_POS_MSEC, (41*60+2)*1000); // skip to 41:42

	cap >> inputFrame;
	resize(inputFrame, inputFrame, Size(), scaleFactor, scaleFactor);
	bgs->Init(inputFrame.size(), foregroundMask);
	
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
		setMouseCallback("OpenCV", MouseEvent, &displayFrame);
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
			
			bgModel = bgs->Substract(inputFrame, foregroundMask);

			if (MEDIAN_SIZE != 0)
				medianBlur(foregroundMask, foregroundMask, MEDIAN_SIZE);
			if (MORPH_SIZE != 0)
				morphologyEx(foregroundMask, foregroundMask, MORPH_OPEN, morphKernel);

			inputFrameNum++;
		}
		
		if (!benchmarkMode)
		{
			hconcat(inputFrame, bgModel, displayFrame);
			Mat tmp;
			hconcat(foregroundMask, foregroundMask, tmp);
			displayFrame.push_back(tmp);
			imshow("OpenCV", displayFrame);
			
			if(record)
				writer << displayFrame;
		}
		
		if (benchmarkMode && inputFrameNum == BENCHMARK_FRAMES_NUM)
			break;

		char key = waitKey(10);
		if(key == 'q')
			break;
		else if (key == ' ')
			loadNew = !loadNew;
	}

	auto t2 = std::chrono::steady_clock::now();
	if (benchmarkMode)
	{
		auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
		std::cout << "Processed " << BENCHMARK_FRAMES_NUM << " inputFrames in " << time_span.count() << " seconds." << std::endl;
	}

	if (record)
		writer.release();

	delete bgs;
	return 0;
}

