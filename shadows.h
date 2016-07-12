#ifndef SHADOWS_H
#define SHADOWS_H

#include "stauffergrimson.h"
#include "gco/GCoptimization.h"
#include <opencv2/opencv.hpp>

using namespace cv;

class Shadows
{
	private:
		const uint8_t thresh1 = 3;
		const uint8_t thresh2 = 7;
		StaufferGrimson* gmm = nullptr;


		struct MySmoothCostFunctor : GCoptimization::SmoothCostFunctor 
		{
			private:
				Mat fgTexture;

			public:
			GCoptimization::EnergyTermType compute(GCoptimization::SiteID s1, GCoptimization::SiteID s2, 
					GCoptimization::LabelID l1, GCoptimization::LabelID l2)
			{
				//std::cout << "s1: " << s1 << ", s2: " << s2 << std::endl;
				//std::cout << "l1: " << l1 << ", l2: " << l2 << std::endl;
				
			//	if (l1 == 1 && l2 == 2)
			//		return 0;
			//	else if(l1 == 2 && l2 == 1)
			//		return 1;
			//	else
			//		return 1;

				//return 1*(fgTexture.at<uint16_t>(s2) - fgTexture.at<uint16_t>(s1));
				
				if ( (l1-l2)*(l1-l2) <= 2 ) return((l1-l2)*(l1-l2));
				else return(2);
			}

			void setFgTexture(const Mat& mat) { fgTexture = mat; }
		};

		MySmoothCostFunctor fct;


	public:
		void Init(StaufferGrimson* gmm);
		void RemoveShadows(InputArray _src, InputArray _bg, OutputArray _dst);
};

#endif
