#include "classifier.h"
#include <vector>

void Classifier::DrawBoundingBoxes(InputOutputArray _frame, InputArray _mask)
{
	Mat frame = _frame.getMat();
	Mat mask = _mask.getMat();

	// find contours in binary mask
	std::vector<Mat> contours;
	findContours(mask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	for (auto& contour: contours)
	{
		if (contourArea(contour) < 300)
			continue;

		RotatedRect rect = minAreaRect(contour);
		Point2f vtx[4];
		rect.points(vtx);
		for(int i = 0; i < 4; i++)
			line(frame, vtx[i], vtx[(i+1)%4], Scalar(255, 0, 255), 2, LINE_AA);
	}
}
