#include "tim.h"

using namespace cv;

int main(int argc, char** argv)
{
    const String keys =
        "{help h usage ? |      | print this message   }"
        "{@file          |<none>| input file           }"
        "{b benchmark    |      | benchmark mode       }"
        "{r record       |      | record output        }";

    CommandLineParser parser(argc, argv, keys);
    parser.about("Tim The Tim");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    String fileName = parser.get<String>(0);
    bool benchmark = parser.has("b");
    bool record = parser.has("r");
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    Tim tim;
    if (tim.open(fileName, benchmark, record))
        tim.processFrames();

    return 0;
}


/* vim: set ft=cpp ts=4 sw=4 sts=4 tw=0 fenc=utf-8 et: */