#include "classifier.h"
#include <vector>
#include <set>

void Classifier::DrawBoundingBoxes(InputOutputArray _frame, InputArray _mask, std::vector<MovingObject>& movingObjects)
{
	Mat frame = _frame.getMat(), mask = _mask.getMat();
	Mat grayFrame;
	cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
	Mat dispFrame = frame.clone();

	for (auto& object: classifiedObjects)
	{
		// if object already contains some tracked points, try to predict their next position
		if (object.prevFeatures.size() > 0)
		{
			std::vector<uint8_t> status;
			std::vector<float>	err;
			object.features.clear();

			//std::vector<Point2f> prevFeatures_offset, features_offset;
			//prevFeatures_offset.resize(object.prevFeatures.size());
			//std::transform(object.prevFeatures.begin(), object.prevFeatures.end(), prevFeatures_offset.begin(),
			//		[&](Point2f& p) { return p + (Point2f)object.selector.tl(); });

			cv::calcOpticalFlowPyrLK(
					prevFrame, grayFrame, // 2 consecutive images
					object.prevFeatures, // input point positions in first im
					object.features, // output point positions in the 2nd
					status,    // tracking success
					err      // tracking error
					);

			//object.features.resize(features_offset.size());
			//std::transform(features_offset.begin(), features_offset.end(), object.features.begin(),
			//		[&](Point2f& p) { return p - (Point2f)object.selector.tl(); });

			std::cout << "ID: " << object.ID << ", status: ";
			for (auto s: status)
				std::cout << (unsigned)s << " ";
			std::cout << ", err: ";
			for (float e: err)
				std::cout << e << ", ";
			std::cout << std::endl;

			// none of points matched, mark object for deletion
			if (std::all_of(err.begin(), err.end(), [](float e) { return e < 2; }))
				object.remove = true;

			// remove points that could not be tracked
			for (int i = 0; i < (int)err.size(); i++)
				if (err[i] < 2) object.features.erase(object.features.begin() + i);
		}
	}

	classifiedObjects.erase(std::remove_if(classifiedObjects.begin(), classifiedObjects.end(),
				[](MovingObject& o) { return o.remove; }), classifiedObjects.end());

	std::vector<MovingObject> objsToAdd;
	for (auto& object: movingObjects)
	{
		auto objMask = object.mask;
		auto contourRect = boundingRect(objMask);
		bool contourMatched = false;
		//rectangle( frame, contourRect, Scalar( 0, 0, 255 ), 2, 1 );		
		//imshow("objMask", 255*objMask);

		// check if predicted feature positions are still within object mask.
		// if not - it's probably a different object.
		// sometimes two or more moving objects match to the same mask,
		// in this case - merge them.
		std::vector<MovingObject*> objectsToMerge;

		for (auto& classifiedObj: classifiedObjects)
		{
			//imshow("obj", 255*obj.mask);
			//waitKey(0);
			uint32_t pointsMatched = 0;
			for (auto& pt: classifiedObj.features)
				pointsMatched += objMask.at<uint8_t>(pt);

			std::cout << "ID: " << classifiedObj.ID << ", matched " << pointsMatched << " out of " 
				<< classifiedObj.features.size() << std::endl;

			if (classifiedObj.features.size() > 0 && pointsMatched >= 0.5 * classifiedObj.features.size())
			{
				contourMatched = true;
				classifiedObj.mask = objMask;
				classifiedObj.minimizeMask();
				std::cout << "ID " << classifiedObj.ID << " matched" << std::endl;
				objectsToMerge.push_back(&classifiedObj);
			}
		}

		if (objectsToMerge.size() > 1)
		{
			std::cout << objectsToMerge.size() << " objects to merge" << std::endl;

			auto obj = MovingObject(frame.size());
			obj.ID = objectsToMerge.front()->ID;

			for (MovingObject* o: objectsToMerge)
			{
				o->remove = true;
				obj.mask += o->mask;
			}
			
			obj.minimizeMask();
			obj.updateTrackedFeatures(grayFrame, frameCounter);
			objsToAdd.push_back(obj);
		}

		if (!contourMatched)
		{
			object.updateTrackedFeatures(grayFrame, frameCounter);
			if (object.features.size() == 0)
				continue;

			object.ID = objCounter++;
			objsToAdd.push_back(object);
		}
	}
	
	for (auto& obj: objsToAdd)
		classifiedObjects.push_back(obj);

	for (auto& obj: classifiedObjects)
	{
		rectangle(dispFrame, obj.selector, Scalar( 255, 0, 0 ), 2, 1 );

		std::string text = std::to_string(obj.ID);
		int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
		double fontScale = 0.6;
		int thickness = 2;

		// center the text
		putText(dispFrame, text, obj.selector.tl(), fontFace, fontScale,
				        Scalar::all(255), thickness, 8);

		for (auto& pt: obj.features)
			circle(dispFrame, pt, 2, Scalar::all(255));

		if (obj.features.size() < 4 || frameCounter - obj.featuresLastUpdated >= 10)
		{
			std::cout << "updating ID " << obj.ID << "..." << std::endl;
			obj.updateTrackedFeatures(grayFrame, frameCounter);
		}

		std::swap(obj.prevFeatures, obj.features);
	}
	
	prevFrame = grayFrame.clone();
	frameCounter++;
	std::cout << "currently " << movingObjects.size() << " objects" << std::endl;
	imshow("lucas", dispFrame);

	//	RotatedRect rect = minAreaRect(contour);
	//	Point2f vtx[4];
	//	rect.points(vtx);
	//	for(int i = 0; i < 4; i++)
	//		line(frame, vtx[i], vtx[(i+1)%4], Scalar(255, 0, 255), 2, LINE_AA);

}
