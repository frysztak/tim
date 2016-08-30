#include "classifier.h"

Classifier::Classifier(const std::vector<Point>& points)
{
	collisionLines[0] = Line(points[0],  points[1]);
	collisionLines[1] = Line(points[2],  points[3]);
}

void Classifier::trackObjects(InputArray _frame, InputArray _mask, std::vector<MovingObject>& movingObjects)
{
	Mat frame = _frame.getMat(), mask = _mask.getMat();
	Mat grayFrame;
	cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

	// predict next position for already recognised objects
	for (auto& object: classifiedObjects)
	{
		if (object.prevFeatures.size() > 0)
			object.predictNextPosition(prevFrame, grayFrame);
	}

	classifiedObjects.erase(std::remove_if(classifiedObjects.begin(), classifiedObjects.end(),
				[](MovingObject& o) { return o.remove; }), classifiedObjects.end());

	// iterate over all detected moving objects and try match them
	// to already known objects saved in 'classifiedObjects'.
	std::vector<MovingObject> objsToAdd;
	for (auto& object: movingObjects)
	{
		auto objMask = object.mask;
		bool objectMatched = false;

		// check if predicted feature positions are still within object mask.
		// if not - it's probably a different object.
		// sometimes two or more moving objects match to the same mask,
		// in this case - merge them.
		std::vector<MovingObject*> objectsToMerge;

		for (auto& classifiedObj: classifiedObjects)
		{
			int pointsMatched = 0;
			for (auto& pt: classifiedObj.features)
				pointsMatched += objMask.at<uint8_t>(pt);

#ifdef DEBUG
			std::cout << "ID: " << classifiedObj.ID << ", matched " << pointsMatched << " out of " 
				<< classifiedObj.features.size() << std::endl;
#endif

			if (pointsMatched >= 0.5 * classifiedObj.features.size())
			{
				objectMatched = true;
				classifiedObj.mask = objMask;
				classifiedObj.minimizeMask();
				objectsToMerge.push_back(&classifiedObj);
#ifdef DEBUG
				std::cout << "ID " << classifiedObj.ID << " matched" << std::endl;
#endif
			}
		}

		// merge objects if necessary
		if (objectsToMerge.size() > 1)
		{
#ifdef DEBUG
			std::cout << objectsToMerge.size() << " objects to merge" << std::endl;
#endif

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

		// in case some object isn't matched with already known objects,
		// add a new one to the list
		if (!objectMatched)
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
		if (obj.features.size() < 4 || frameCounter - obj.featuresLastUpdated >= 10)
			obj.updateTrackedFeatures(grayFrame, frameCounter);

		std::swap(obj.prevFeatures, obj.features);
	}
	
	prevFrame = grayFrame.clone();
	frameCounter++;
}

void Classifier::checkCollisions(std::vector<MovingObject>& objects)
{
	collisionLines[0].intersect(objects);
	collisionLines[1].intersect(objects);
}

void Classifier::drawBoundingBoxes(InputOutputArray _frame)
{
	Mat frame = _frame.getMat();
	
	for (auto& obj: classifiedObjects)
	{
		rectangle(frame, obj.selector, Scalar(255, 0, 0), 2, 1);

		std::string text = std::to_string(obj.ID);
		int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
		double fontScale = 0.5;
		int thickness = 2;

		// center the text
		putText(frame, text, obj.selector.tl(), fontFace, fontScale,
		        Scalar::all(255), thickness, 8);

		for (auto& pt: obj.features)
			circle(frame, pt, 3, Scalar(255, 0, 255));
	}

	//	RotatedRect rect = minAreaRect(contour);
	//	Point2f vtx[4];
	//	rect.points(vtx);
	//	for(int i = 0; i < 4; i++)
	//		line(frame, vtx[i], vtx[(i+1)%4], Scalar(255, 0, 255), 2, LINE_AA);
}

void Classifier::drawCollisionLines(InputOutputArray _frame)
{
	Mat frame = _frame.getMat();
	
	collisionLines[0].draw(frame);
	collisionLines[1].draw(frame);
}
