#include "classifier.h"
#include <vector>

void Classifier::DrawBoundingBoxes(InputOutputArray _frame, InputArray _mask, InputArray _roiMask)
{
	Mat frame = _frame.getMat(), mask = _mask.getMat(), roiMask = _roiMask.getMat();

	// find contours in binary mask
	std::vector<Mat> contours;
	findContours(mask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	// for already detected objects: predict next position.
	// if prediction fails, remove object.
	objects.erase(std::remove_if(objects.begin(), objects.end(), 
				[&](Object& obj) 
				{
					if (!obj.tracker->update(frame, obj.rect))
						return true; // update failed
					// update succeded, test if predicted points lies within roi polygon
					auto tl = obj.rect.tl();
					auto br = obj.rect.br();
					if (roiMask.at<uint8_t>(tl) == 0 || roiMask.at<uint8_t>(br) == 0)
						return true;
					return false;
				}), objects.end());

	for (auto& contour: contours)
	{
		if (contourArea(contour) < 300)
			continue;

		auto contourRect = boundingRect(contour);
		bool contourMatched = false;
		//rectangle( frame, contourRect, Scalar( 0, 0, 255 ), 2, 1 );

		for (auto& obj: objects)
		{
			if (obj.doRectsOverlap(contourRect))
			{
				// we found a bigger rect
				obj.tracker = Tracker::create(trackerType);
				obj.tracker->init(frame, contourRect);
				obj.rect = contourRect;
				contourMatched = true;
				break;
			}
		}

		if (!contourMatched)
		{
			auto obj = Object();
			obj.tracker = Tracker::create(trackerType);
			obj.tracker->init(frame, contourRect);
			obj.rect = contourRect;
			obj.ID = objCounter++;
			objects.push_back(obj);
		}
	}
	
	for (auto& obj: objects)
	{
		rectangle( frame, obj.rect, Scalar( 255, 0, 0 ), 2, 1 );

		std::string text = std::to_string(obj.ID);
		int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
		double fontScale = 0.6;
		int thickness = 2;

		// center the text
		putText(frame, text, obj.rect.tl(), fontFace, fontScale,
				        Scalar::all(255), thickness, 8);
	}

	//	RotatedRect rect = minAreaRect(contour);
	//	Point2f vtx[4];
	//	rect.points(vtx);
	//	for(int i = 0; i < 4; i++)
	//		line(frame, vtx[i], vtx[(i+1)%4], Scalar(255, 0, 255), 2, LINE_AA);

}
