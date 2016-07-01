#include <iostream>
#include <string>
#include "args.hxx"
#include "tim.h"

using namespace cv;
using namespace args;

int main(int argc, char** argv)
{
	Tim tim;

	ArgumentParser parser("Tim the Tim");
	HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
	Positional<std::string> input(parser, "stream", "Input video stream");
	Flag benchmark(parser, "benchmark", "Run in benchmark mode", {"b", "benchmark"});
	Flag record(parser, "record", "Record output", {"r", "record"});

	try
	{
		parser.ParseCLI(argc, argv);
	}
	catch (args::Help)
	{
		std::cout << parser;
		return 0;
	}
	catch (args::ParseError e)
	{
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return 1;
	}

	std::string inputStream;
	if(!input)
	{
		std::cout << "you need to specify input stream. bye." << std::endl;
		return 0;
	}
	else
	{
		inputStream = args::get(input);
	}

	if (tim.open(inputStream, args::get(benchmark), args::get(record)))
		tim.processFrames();

	//cap.set(CAP_PROP_POS_MSEC, 1292000); // skip to 21:30
	//cap.set(CAP_PROP_POS_MSEC, (41*60+2)*1000); // skip to 41:42

	/*
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

	
	auto t1 = std::chrono::steady_clock::now();
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
	*/
	return 0;
}

