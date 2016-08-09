#ifndef BACKGROUND_H
#define BACKGROUND_H

#include <opencv2/opencv.hpp>

#define GAUSSIANS_PER_PIXEL 3

using namespace cv;

class Background
{
	struct Gaussian 
	{
		float meanB;
		float meanG;
		float meanR;
		float variance = 0; // it's used to determine if Gaussian was already used or not. see findMixtureSize().
		float weight;

		bool operator>(const Gaussian& other) const
		{
			return (this->weight/sqrt(this->variance)) > (other.weight/sqrt(other.variance));
		}
	};

	typedef Gaussian GaussianMixture[GAUSSIANS_PER_PIXEL];
	
	public:
		Background();
		void init(const Size& size);
		void processFrame(InputArray _src, OutputArray _foregroundMask);
		const Mat& getCurrentBackground() const;
		const Mat& getCurrentStdDev() const;

	private:
		const float initialVariance;
		const float initialWeight;
		const float learningRate;
		const float foregroundThreshold;
		
		const float etaConst;

		Mat currentBackground, currentStdDev;
		GaussianMixture *gaussians = nullptr;
		int findMixtureSize(const GaussianMixture& mixture) const;

		bool processPixel(const Vec3b& rgb, GaussianMixture& mixture);
};

#endif
