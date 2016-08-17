#include "classifier.h"
#include <vector>
#include <set>

void Classifier::DrawBoundingBoxes(InputOutputArray _frame, InputArray _mask, InputArray _roiMask)
{
	Mat frame = _frame.getMat(), mask = _mask.getMat(), roiMask = _roiMask.getMat();
	Mat grayFrame;
	cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
	Mat dispFrame = frame.clone();

	// find contours in binary mask
	Mat labels;
	int nLabels = connectedComponents(mask, labels, 8, CV_16U);
	std::vector<Mat> objMasks;

	for (int i = 1; i < nLabels; i++)
		objMasks.push_back(Mat::zeros(frame.size(), CV_8U));

	for (int i = 0; i < labels.rows * labels.cols; i++)
	{
		auto v = labels.at<uint16_t>(i);
		if (v == 0) continue;
		objMasks[v-1].at<uint8_t>(i) = 1;
	}

	auto it = std::remove_if(objMasks.begin(), objMasks.end(), 
			[](Mat& mask) { return countNonZero(mask) < 150; });
	objMasks.erase(it, objMasks.end());

	// for already detected objects: predict next position.
	// if prediction fails, remove object.
	objects.erase(std::remove_if(objects.begin(), objects.end(), 
				[&](Object& obj) 
				{
					std::vector<uint8_t> status;
					std::vector<float>	err;
					obj.features.clear();

					cv::calcOpticalFlowPyrLK(
							prevFrame, grayFrame, // 2 consecutive images
							obj.prevFeatures, // input point positions in first im
							obj.features, // output point positions in the 2nd
							status,    // tracking success
							err      // tracking error
							);
					std::cout << "ID: " << obj.ID << ", status: ";
					for (auto s: status)
						std::cout << (unsigned)s << " ";
					std::cout << ", err: ";
					for (float e: err)
						std::cout << e << ", ";
					std::cout << std::endl;

					if (std::all_of(err.begin(), err.end(), [](float e) { return e < 2; }))
						return true;

					// remove points that could not be tracked
					for (int i = 0; i < (int)err.size(); i++)
						if (err[i] < 2) obj.features.erase(obj.features.begin() + i);

					// update succeded, test if predicted points lies within roi polygon
					//auto tl = obj.rect.tl();
					//auto br = obj.rect.br();
					//if (roiMask.at<uint8_t>(tl) == 0 || roiMask.at<uint8_t>(br) == 0)
					//	return true;
					return false;
				}), objects.end());

	for (auto& objMask: objMasks)
	{
		auto contourRect = boundingRect(objMask);
		bool contourMatched = false;
		//rectangle( frame, contourRect, Scalar( 0, 0, 255 ), 2, 1 );
		
		//imshow("objMask", 255*objMask);

		std::vector<Object> objectsToMerge;
		for (auto& obj: objects)
		{
			//imshow("obj", 255*obj.mask);
			//waitKey(0);
			auto b = (rand() % 150) + 155, r = (rand() % 150) + 155, g = (rand() % 150) + 155;
			for (auto& pt: obj.features)
				circle(dispFrame, pt, 2, Scalar(b,g,r));

			uint32_t pointsMatched = 0;
			for (auto& pt: obj.features)
				pointsMatched += objMask.at<uint8_t>(pt);

			std::cout << "ID: " << obj.ID << ", matched " << pointsMatched << " out of " << obj.features.size() << std::endl;

			if (pointsMatched >= 0.5 * obj.features.size())
			{
				contourMatched = true;
				obj.mask = objMask;
				std::cout << "ID " << obj.ID << " matched" << std::endl;
				objectsToMerge.push_back(obj);
			}
		}

		if (objectsToMerge.size() > 1)
		{
			std::cout << objectsToMerge.size() << " objects to merge" << std::endl;

			std::vector<Point2f> features;
			std::vector<Point2i> featuresInt;
			for (auto& obj: objectsToMerge)
			{
				//imshow("to merge", 255*obj.mask);
				waitKey(0);

				for (auto& pt: obj.features)
					features.push_back(pt);
			}

			Mat(features).convertTo(featuresInt, Mat(featuresInt).type());

			// remove duplicates
			std::sort(featuresInt.begin(), featuresInt.end(),
					[](Point2i& lhs, Point2i& rhs) { return lhs.x < rhs.x && lhs.y < rhs.y; });
			featuresInt.erase(std::unique(featuresInt.begin(), featuresInt.end()), featuresInt.end());
			features.clear();

			// convert back to float
			Mat(featuresInt).convertTo(features, Mat(features).type());

			auto obj = Object();
			obj.ID = objectsToMerge.front().ID;
			obj.mask = Mat::zeros(frame.size(), CV_8U); 
			obj.featuresLastUpdated = frameCounter;

			for (auto& o: objectsToMerge)
			{
				objects.erase(std::remove(objects.begin(), objects.end(), o), objects.end());
				obj.mask += o.mask;
			}
			//imshow("merged mask", 255*obj.mask);
			
			features.clear();
			// update features
			cv::goodFeaturesToTrack(grayFrame, // the image 
					features,   // the output detected features
					10,  // the maximum number of features 
					0.01,     // quality level
					8, // min distance between two features
					obj.mask	
					);

			obj.features = features;
			objects.push_back(obj);
		}

		if (!contourMatched)
		{
			auto obj = Object();
			Mat mask = Mat::zeros(frame.size(), CV_8UC1);
			mask(contourRect) = 255;

			cv::goodFeaturesToTrack(grayFrame, // the image 
					obj.features,   // the output detected features
					10,  // the maximum number of features 
					0.01,     // quality level
					8, // min distance between two features
					mask	
					);
			if (obj.features.size() == 0)
				continue;
			obj.rect = contourRect;
			obj.ID = objCounter++;
			obj.mask = objMask;
			obj.featuresLastUpdated = frameCounter;
			objects.push_back(obj);
		}
	}
	
	for (auto& obj: objects)
	{
		auto r = boundingRect(obj.features);
		rectangle( dispFrame, r, Scalar( 255, 0, 0 ), 2, 1 );

		std::string text = std::to_string(obj.ID);
		int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
		double fontScale = 0.6;
		int thickness = 2;

		// center the text
		putText(dispFrame, text, r.tl(), fontFace, fontScale,
				        Scalar::all(255), thickness, 8);

		//if (frameCounter - obj.featuresLastUpdated >= 15)
		if (obj.features.size() < 5)
		{
			std::vector<Point2f> newFeatures;
			cv::goodFeaturesToTrack(grayFrame, // the image 
					newFeatures,   // the output detected features
					10,  // the maximum number of features 
					0.01,     // quality level
					8, // min distance between two features
					obj.mask	
					);

			if (newFeatures.size() > 1)
			{
				std::swap(obj.features, obj.prevFeatures);
				obj.features = newFeatures;
				obj.featuresLastUpdated = frameCounter;
			}
		}

		std::swap(obj.prevFeatures, obj.features);
	}
	
	prevFrame = grayFrame.clone();
	frameCounter++;
	std::cout << "currently " << objects.size() << " objects" << std::endl;
	imshow("lucas", dispFrame);

	//	RotatedRect rect = minAreaRect(contour);
	//	Point2f vtx[4];
	//	rect.points(vtx);
	//	for(int i = 0; i < 4; i++)
	//		line(frame, vtx[i], vtx[(i+1)%4], Scalar(255, 0, 255), 2, LINE_AA);

}
