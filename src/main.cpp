#include "tim.h"

using namespace cv;

int main(int argc, char** argv)
{
    const String keys =
        "{help h usage ? |      | print this message              }"
        "{@file          |<none>| input file                      }"
        "{b benchmark    |      | benchmark mode                  }"
        "{r record       |      | record output                   }"
        "{dnt            |      | don't track moving objects      }"
        "{cc colours     |      | classify colours of passing objects"
        " (more experimental and broken than anything else in this application) }";

    CommandLineParser parser(argc, argv, keys);
    parser.about("Tim The Tim");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    Tim tim;
    TimParameters params = 
    {
        .fileName = parser.get<String>(0), 
        .benchmark = parser.has("b"),
        .record = parser.has("r"),
        .classifyColours = parser.has("cc"),
        .dontTrack = parser.has("dnt"),
    };

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    if (tim.open(params))
        tim.processFrames();

    return 0;
}


/* vim: set ft=cpp ts=4 sw=4 sts=4 tw=0 fenc=utf-8 et: */
