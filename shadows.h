#ifndef SHADOWS_H
#define SHADOWS_H

#include "gco/GCoptimization.h"
#include <opencv2/opencv.hpp>

using namespace cv;

class Shadows
{
	private:
		const uint8_t distanceThreshold;
		const uint8_t absoluteThreshold;

		struct MySmoothCostFunctor : GCoptimization::SmoothCostFunctor
		{
			GCoptimization::EnergyTermType compute(GCoptimization::SiteID s1, GCoptimization::SiteID s2, 
					GCoptimization::LabelID l1, GCoptimization::LabelID l2);
		};

		MySmoothCostFunctor smoothFunctor;

	public:
		Shadows();
		void removeShadows(InputArray _src, InputArray _bg, InputArray _bgTexture, InputArray _fgMask, OutputArray _dst);
};

#endif
