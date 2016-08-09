#ifndef BACKGROUND_H
#define BACKGROUND_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

class Background
{
	struct Gaussian 
	{
		float meanB;
		float meanG;
		float meanR;
		float variance;
		float weight;

		bool operator>(const Gaussian& other) const
		{
			return (this->weight/sqrt(this->variance)) > (other.weight/sqrt(other.variance));
		}
	};

	typedef std::vector<Gaussian> GaussianMixture;
	
	public:
		Background();
		void init(const Size& size);
		void processFrame(InputArray _src, OutputArray _foregroundMask);
		const Mat& getCurrentBackground() const;
		const Mat& getCurrentStdDev() const;

	private:
		const float initialVariance;
		const float initialWeight;
		const int gaussiansPerPixel;
		const float learningRate;
		const float foregroundThreshold;
		
		const float etaConst;

		Mat currentBackground, currentStdDev;
		std::vector<GaussianMixture> gaussians;

		bool processPixel(const Vec3b& rgb, GaussianMixture& mixture);
};

#endif
