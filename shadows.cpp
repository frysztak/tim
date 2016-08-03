#include "shadows.h"
#include "background.h"

Shadows::Shadows(ShadowsParameters& params) : params(params) 
{
}

void Shadows::removeShadows(InputArray _src, InputArray _bg, InputArray _bgStdDev, InputArray _fgMask, OutputArray _dst)
{
	Mat frame = _src.getMat(), background = _bg.getMat(), backgroundStdDev = _bgStdDev.getMat(),
		foregroundMask = _fgMask.getMat(), labelMask = _dst.getMat(), 
		shadowMask = Mat::zeros(frame.size(), CV_8U);

	// object masks: segment foreground mask into separate moving movingObjects
	int nLabels = connectedComponents(foregroundMask, objectLabels, 8, CV_16U);
	std::vector<MovingObject> movingObjects;
	for (int label = 0; label < nLabels; label++)
		movingObjects.emplace_back(objectLabels.size());
	
	for (int idx = 0; idx < objectLabels.rows*objectLabels.cols; idx++)
	{
		uint16_t label = objectLabels.at<uint16_t>(idx);
		if (label == 0) continue;

		movingObjects[label].mask.at<uint8_t>(idx) = 1;
	}

	// remove tiniest movingObjects	
	auto it = std::remove_if(movingObjects.begin(), movingObjects.end(), 
			[&](MovingObject& object) { return countNonZero(object.mask) < params.minObjectSize; });
	movingObjects.erase(it, movingObjects.end());

	for (auto& obj: movingObjects)
		minimizeObjectMask(obj);

	// calculate D (eq. 7) 
	D = Mat::zeros(frame.size(), CV_32FC3);
	Mat tmp = Mat::zeros(frame.size(), CV_32FC3);
	const double v = 1;
	background.convertTo(D, CV_32F, 1, v);
	frame.convertTo(tmp, CV_32F, 1, v);
	divide(D, tmp, tmp, 1, CV_32FC3);
	D *= 0;
	tmp.copyTo(D, foregroundMask);
	//D *= 0.25;
#if DEBUG
	double maxVal, minVal;
	minMaxLoc(D, &minVal, &maxVal);
	//std::cout << "D mean: " << cv::mean(D, foregroundMask) << std::endl;
	//imshow("D", D/maxVal);
#endif

	// divide each moving objects into sub-segments
	for (MovingObject& object: movingObjects)
	{
		if (params.edgeCorrection)
			erode(object.mask, object.mask, getStructuringElement(MORPH_RECT, Size(5,5)));

		float grThr = params.gradientThreshold;
		if (params.autoGradientThreshold)
		{
			// calculate gradient threshold
			float objSize = countNonZero(object.mask);
			Mat objBg, objStdDev;
			background(object.selector).copyTo(objBg, object.mask);
			backgroundStdDev(object.selector).copyTo(objStdDev, object.mask);

			Scalar meanSum = cv::sum(objBg);
			Scalar stdDevSum = cv::sum(objStdDev);

			grThr = params.alpha / objSize;
			grThr *= meanSum[0] * stdDevSum[0] / objSize; 
			grThr *= params.gradientThresholdMultiplier;
#if DEBUG
			std::cout << "threshold: " << grThr << ", obj size: " << objSize << std::endl;
#endif
		}

		Mat segmentLabels = Mat::zeros(object.mask.size(), CV_16U);
		int currentLabel = 0;
		
		for (int r = 0; r < object.mask.rows; r++)
		{
			for (int c = 0; c < object.mask.cols; c++)
			{
				if (object.mask.at<uint8_t>(r, c) == 0 || segmentLabels.at<uint16_t>(r, c) != 0)
					continue;
				
				findSegment(object, Point(r, c), segmentLabels, ++currentLabel, grThr);
			}
		}

		object.segmentLabels = segmentLabels;
	}

	// at this point, each moving object is divided into segments and labeled.

	Mat globalSegmentMap = Mat::zeros(frame.size(), CV_16U);
	uint32_t globalSegmentCounter = 0;

	if (params.edgeCorrection)
		erode(objectLabels, objectLabels, getStructuringElement(MORPH_RECT, Size(5,5)));

#if DEBUG
	Mat luminanceCritetion = Mat::zeros(frame.size(), CV_8U);
	Mat sizeCriterion = Mat::zeros(frame.size(), CV_8U);
	Mat externalPointsCriterion = Mat::zeros(frame.size(), CV_8U);
#endif

	for (MovingObject& object: movingObjects)
	{
		auto& segmentLabels = object.segmentLabels;
		const auto selector = object.selector;
		globalSegmentMap(selector) += segmentLabels;

		for (Segment& segment: object.segments)
		{
			globalSegmentCounter++;

			// luminance criterion  (eq. 10)
			Scalar mean = cv::mean(D(selector), segment.mask);
			bool luminance_ok = (mean[0] > params.luminanceThreshold) && (mean[1] > params.luminanceThreshold) && 
				(mean[2] > params.luminanceThreshold);
			if (!luminance_ok)
			{
				// it's surely foreground
				shadowMask(selector).setTo(2, segment.mask);
				continue;
			}
#if DEBUG
			else
				luminanceCritetion(selector).setTo(1, segment.mask);
#endif

			// size criterion (eq. 11)
			bool size_ok = segment.area > params.lambda * countNonZero(object.mask);
#if DEBUG
			if (size_ok)
				sizeCriterion(selector).setTo(1, segment.mask);
#endif

			// calculate number of internal and external terminal points
			int nExternal = 0, nAll = 0;
			auto smallObjectLabels = objectLabels(selector);

			for (int r = 0; r < segment.mask.rows; r++)
			{
				for (int c = 0; c < segment.mask.cols; c++)
				{
					if (segment.mask.at<uint8_t>(r, c) == 0) continue;

					uint16_t segmentLabel = segmentLabels.at<uint16_t>(r,c);
					uint16_t objectLabel = objectLabels.at<uint16_t>(r,c);
					
					// first check if we're at the edge of label
					if ((r < segmentLabels.rows - 1 && (segmentLabel != segmentLabels.at<uint16_t>(r+1,c))) ||
						(c < segmentLabels.cols - 1 && (segmentLabel != segmentLabels.at<uint16_t>(r,c+1))) ||
						(r > 0 && (segmentLabel != segmentLabels.at<uint16_t>(r-1,c))) ||
						(c > 0 && (segmentLabel != segmentLabels.at<uint16_t>(r,c-1))))
					{
						// we are.
						nAll++;

						// check if we're at external point
						if ((r < smallObjectLabels.rows - 1 && (objectLabel != smallObjectLabels.at<uint16_t>(r+1,c))) ||
							(c < smallObjectLabels.cols - 1 && (objectLabel != smallObjectLabels.at<uint16_t>(r,c+1))) ||
							(r > 0 && (objectLabel != smallObjectLabels.at<uint16_t>(r-1,c))) ||
							(c > 0 && (objectLabel != smallObjectLabels.at<uint16_t>(r,c-1))))
						//if ((r < smallObjectLabels.rows - 1 && (0 == smallObjectLabels.at<uint16_t>(r+1,c))) ||
						//	(c < smallObjectLabels.cols - 1 && (0 == smallObjectLabels.at<uint16_t>(r,c+1))) ||
						//	(r > 0 && (0 == smallObjectLabels.at<uint16_t>(r-1,c))) ||
						//	(c > 0 && (0 == smallObjectLabels.at<uint16_t>(r,c-1))))

						{
							nExternal++;
						}
					}
				}
			}

			// extrinsic terminal point criterion (eq. 12)
			bool extrinsic_ok = (nExternal / float(nAll)) > params.tau;
#if DEBUG
			if (extrinsic_ok)
				externalPointsCriterion(selector).setTo(1, segment.mask);
#endif

			if (luminance_ok && size_ok && extrinsic_ok)
				shadowMask(selector).setTo(1, segment.mask);
		}
	}

#if DEBUG
	imshow("luminanceCritetion", 255*luminanceCritetion);
	imshow("sizeCriterion", 255*sizeCriterion);
	imshow("externalPointsCriterion", 255*externalPointsCriterion);
	
	if(globalSegmentCounter != 0)
		showSegmentation(globalSegmentCounter, globalSegmentMap);

	imshow("shadowMask w/ blanks", (255/2)*shadowMask);
#endif

	fillInBlanks(foregroundMask, shadowMask);
	//showSegmentation(nLabels, objectLabels);
	
	shadowMask.copyTo(_dst);
}

void Shadows::findSegment(MovingObject& object, Point startPoint, InputOutputArray _segmentLabels, 
		 uint16_t label, float gradientThreshold)
{
	Mat labels = _segmentLabels.getMat(), objectMask = object.mask, 
		segmentMask = Mat::zeros(objectMask.size(), CV_8U),
		visited = Mat::zeros(objectMask.size(), CV_8U); 

	std::vector<Point> stack;
	int area = 0;
	Mat D_roi = D(object.selector);

	// remember points in case segment turns out to be too small
	std::vector<Point> visitedPoints; 

	stack.push_back(startPoint);
	while(!stack.empty())
	{
		auto checkPoint = [&](Point p)
		{
			Vec3f ratio1 = D_roi.at<Vec3f>(startPoint.x, startPoint.y);
			Vec3f ratio2 = D_roi.at<Vec3f>(p.x, p.y);

			if (gradientThreshold > abs(ratio1[0] - ratio2[0]) &&
				gradientThreshold > abs(ratio1[1] - ratio2[1]) &&
				gradientThreshold > abs(ratio1[2] - ratio2[2]))
			{
				stack.push_back(p);
				visitedPoints.push_back(p);
				return true;
			}
			return false;
		};
		
		Point p1 = stack[stack.size() - 1]; 

		visited.at<uint8_t>(p1.x, p1.y) = 1;
		labels.at<uint16_t>(p1.x, p1.y) = label;
		segmentMask.at<uint8_t>(p1.x, p1.y) = 1;
		area++;
		
		if (p1.x < labels.rows - 1 && !visited.at<uint8_t>(p1.x+1, p1.y) && objectMask.at<uint8_t>(p1.x+1,p1.y) != 0)
			if(checkPoint(Point(p1.x+1, p1.y))) continue;
		if (p1.y < labels.cols - 1 && !visited.at<uint8_t>(p1.x,p1.y+1) && objectMask.at<uint8_t>(p1.x,p1.y+1) != 0)
			if(checkPoint(Point(p1.x, p1.y+1))) continue;
		if (p1.x > 0 && !visited.at<uint8_t>(p1.x-1, p1.y) && objectMask.at<uint8_t>(p1.x-1,p1.y) != 0)
			if(checkPoint(Point(p1.x-1, p1.y))) continue;
		if (p1.y > 0 && !visited.at<uint8_t>(p1.x,p1.y-1) && objectMask.at<uint8_t>(p1.x,p1.y-1) != 0)
			if(checkPoint(Point(p1.x, p1.y-1))) continue;

		stack.pop_back();
	}

	if (area < params.minSegmentSize)
	{
		// area is too small, so revert labelling at every visited pixel
		for (auto& p: visitedPoints)
			labels.at<uint16_t>(p.x, p.y) = 0;
	}
	else
		object.segments.emplace_back(segmentMask, area);
}

void Shadows::fillInBlanks(InputArray _fgMask, InputArray _mask)
{
	Mat fgMask = _fgMask.getMat(), mask = _mask.getMat();

	auto findDistance = [&](int deltaR, int deltaC, int startR, int startC)
	{
		int r = startR, c = startC;
		int distance = 0;
		int value = 0;

		while(r >= 0 && c >= 0 && r <= mask.rows && c <= mask.cols)
		{
			if (mask.at<uint8_t>(r,c) != 0)
			{
				value = mask.at<uint8_t>(r,c);
				break;
			}

			r += deltaR;
			c += deltaC;
			distance++;
		}

		return std::make_tuple(distance, value);
	};

	for (int r = 0; r < fgMask.rows; r++)
	{
		for (int c = 0; c < fgMask.cols; c++)
		{
			if (fgMask.at<uint8_t>(r,c) == 0 || (fgMask.at<uint8_t>(r,c) != 0 && mask.at<uint8_t>(r,c) != 0))
				continue;

			auto right = findDistance(1, 0, r, c);
			auto left = findDistance(-1, 0, r, c);
			auto up = findDistance(0, 1, r, c);
			auto down = findDistance(0, -1, r, c);

			std::vector<decltype(right)> pairs = { right, left, up, down };
			auto it = std::min_element(pairs.begin(), pairs.end(), [](decltype(right)& a, decltype(right)& b)
					{ return std::get<0>(a) < std::get<0>(b); });

			mask.at<uint8_t>(r,c) = std::get<1>(*it);
		}
	}
}

void Shadows::showSegmentation(int nSegments, InputArray _labels)
{
	Mat labels = _labels.getMat();
	Mat segmentLabelsColour = Mat::zeros(labels.size(), CV_8UC3);
	std::vector<Vec3b> colors(nSegments);
	colors[0] = Vec3b(0, 0, 0);//background
	for(int label = 1; label < nSegments; ++label)
		colors[label] = Vec3b( (rand()&205) + 50, (rand()&205) + 50, (rand()&205) + 50);
	for (int idx = 0; idx < segmentLabelsColour.cols * segmentLabelsColour.rows; idx++)
	{
		int label = labels.at<uint16_t>(idx);
		segmentLabelsColour.at<Vec3b>(idx) = colors[label];
	}
	imshow("segments", segmentLabelsColour);
}

void Shadows::minimizeObjectMask(MovingObject& obj)
{
	Rect rect = boundingRect(obj.mask);
	Size newSize = rect.size();
	auto offset = rect.tl();

	obj.selector = Rect(offset.x, offset.y, newSize.width, newSize.height);
	obj.mask = obj.mask(obj.selector);
}
