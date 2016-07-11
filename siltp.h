#ifndef SILTP_H
#define SILTP_H

#include <opencv2/opencv.hpp>

using namespace cv;

void SILTP_8x1(InputArray _src, OutputArray _dst);
void SILTP_16x2(InputArray _src, OutputArray _dst);
uint8_t hamming_distance(uint32_t a, uint32_t b);

#endif
