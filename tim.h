#ifndef TIM_H
#define TIM_H

#include "benedek.h"
#include "classifier.h"
#include <string>

#define BENCHMARK_FRAMES_NUM 400

using namespace std;

class Tim
{
	public:
		bool open(const string& name, bool benchmark, bool record);
		void processFrames();

	private:
		const string dataRootDir = "/mnt/things/tim/";
		const double scaleFactor = .5;

		uint medianFilterSize = 0;
		uint morphFilterSize = 0;
		bool paused = false;
		bool benchmarkMode = false;
		bool record = false;
		uint32_t frameCount = 0;
		Mat morphKernel;

		BenedekSziranyi benedek;
		Classifier classifier;
		VideoCapture videoCapture;
		VideoWriter videoWriter;
		Size frameSize;
};

#endif
