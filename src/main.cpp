#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include "stauffergrimson.h"

using namespace cv;

const float scaleFactor = .5f;

const std::string videoPath = "/mnt/things/car detection/videos/sofia.mp4";
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
	VideoWriter writer;

	VideoCapture cap(videoPath);
	if (!cap.isOpened())
	{
		std::cout << "Could not open video file" << std::endl;
		return -1;
	}
	//cap.set(CAP_PROP_POS_MSEC, 1292000); // skip to 21:30
	cap.set(CAP_PROP_POS_MSEC, (41*60+2)*1000); // skip to 41:42

	Mat inputFrame, foregroundMask, show, bgModel;
	cap >> inputFrame;
	resize(inputFrame, inputFrame, Size(), scaleFactor, scaleFactor);
	bgs->Init(inputFrame.size());
	foregroundMask = inputFrame.clone();
	
	bool benchmarkMode = false;
	bool record = false;
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
		setMouseCallback("OpenCV", MouseEvent, &show);
	}
	int morph_size = 2;
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(morph_size, morph_size));

	uint32_t inputFrameNum = 0;
	uint16_t inputFramesToProcess = 400;
	auto t1 = std::chrono::steady_clock::now();

	bool loadNew = true;
	while(true)
	{
		if(loadNew)
		{
			cap >> inputFrame;
			resize(inputFrame, inputFrame, Size(), scaleFactor, scaleFactor);
			
			inputFrameNum++;

			bgModel = bgs->Substract(inputFrame, foregroundMask);

			medianBlur(foregroundMask, foregroundMask, 3);
			morphologyEx(foregroundMask, foregroundMask, MORPH_OPEN, kernel);
		}
		
		if (!benchmarkMode)
		{
			show = inputFrame;
			show.push_back(foregroundMask);
			imshow("OpenCV", show);
			
			if(record)
				writer << show;
		}
		
		if (benchmarkMode && inputFrameNum == inputFramesToProcess)
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
		std::cout << "Processed " << inputFramesToProcess << " inputFrames in " << time_span.count() << " seconds." << std::endl;
	}

	if (record)
		writer.release();

	delete bgs;
	return 0;
}

