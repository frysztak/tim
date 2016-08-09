#include "background.h"
#include <cstdlib>

Background::Background() :
	initialVariance(20),
	initialWeight(0.05),
	learningRate(0.05),
	foregroundThreshold(8.5),
	etaConst(pow(2 * M_PI, 3.0 / 2.0))
{
}

void Background::init(const Size& size)
{
	posix_memalign((void**)&gaussians, 16, size.area() * sizeof(GaussianMixture));

	currentBackground = Mat::zeros(size, CV_8UC3);
	currentStdDev = Mat::zeros(size, CV_32F);
}

Background::~Background()
{
	free(gaussians);
}

void Background::processFrame(InputArray _src, OutputArray _foregroundMask)
{
	Mat src = _src.getMat(), foregroundMask = _foregroundMask.getMat();

	for (int row = 0; row < src.rows; ++row)
	{
		uint8_t *foregroundMaskPtr = foregroundMask.ptr<uint8_t>(row);
		uint8_t *currentBackgroundPtr = currentBackground.ptr<uint8_t>(row);
		float *currentStdDevPtr = currentStdDev.ptr<float>(row);
		uint8_t *srcPtr = src.ptr<uint8_t>(row);
		int idx = src.cols * row; 

		for (int col = 0; col < src.cols; col++, idx++)
		{
			Vec3b bgr;
			bgr[0] = *srcPtr++; 
			bgr[1] = *srcPtr++; 
			bgr[2] = *srcPtr++; 
				
			GaussianMixture& mog = gaussians[idx];
			bool foreground = processPixel(bgr, mog);
			*foregroundMaskPtr++ = foreground ? 1 : 0;

			// update current background model (or rather, background image)
			auto& gauss = mog[0];
			*currentBackgroundPtr++ = gauss.meanB;
			*currentBackgroundPtr++ = gauss.meanG;
			*currentBackgroundPtr++ = gauss.meanR;
			*currentStdDevPtr++ = sqrt(gauss.variance);
		}
	}

	medianBlur(foregroundMask, foregroundMask, 3);
}

bool Background::processPixel(const Vec3b& bgr, GaussianMixture& mixture)
{
	double weightSum = 0.0;
	bool matchFound = false;

	for (Gaussian& gauss : mixture)
	{
		float dB = gauss.meanB - bgr[0];
		float dG = gauss.meanG - bgr[1];
		float dR = gauss.meanR - bgr[2];
		float distance = dR*dR + dG*dG + dB*dB;

		if (sqrt(distance) < 2.5*sqrt(gauss.variance) && !matchFound)
		{
			matchFound = true;
			
			// determinant of covariance matrix (eq. 4 in Stauffer&Grimson's paper) equals to sigma^6. 
			// we need sigma^3, let's so calculate square root of variance (stdDev) and multiply it 3 times.
			float stdDev = sqrt(gauss.variance);

			float exponent = (-0.5 * distance) / gauss.variance;
			float eta = exp(exponent) / (etaConst * stdDev * stdDev * stdDev);

			float rho = learningRate * eta;
			float oneMinusRho = 1.0 - rho;

			gauss.meanB = oneMinusRho*gauss.meanB + rho*bgr[0];
			gauss.meanG = oneMinusRho*gauss.meanG + rho*bgr[1];
			gauss.meanR = oneMinusRho*gauss.meanR + rho*bgr[2];
			gauss.variance = oneMinusRho*gauss.variance + rho*distance;
		}
		else
		{
			gauss.weight = (1.0 - learningRate)*gauss.weight;
		}

		weightSum += gauss.weight;
	}

	if(!matchFound)
	{
		// pixel didn't match any of currently existing Gaussians.
		int mixtureSize = findMixtureSize(mixture);
		if(mixtureSize < GAUSSIANS_PER_PIXEL)
		{
			// add new Gaussian to the list
			mixture[mixtureSize] = Gaussian();
		}
		else 
		{
			// we can't add another Gaussian.
			// as per paper, let's modify least probable distribution.
			// but first, we have to sort by (weight/variance) parameter.
			std::sort(std::begin(mixture), std::end(mixture), std::greater<Gaussian>());
			mixtureSize = GAUSSIANS_PER_PIXEL - 1;
		}

		Gaussian& gauss = mixture[mixtureSize];

		gauss.meanB = bgr[0];
		gauss.meanG = bgr[1];
		gauss.meanR = bgr[2];	
		gauss.weight = initialWeight;
		gauss.variance = initialVariance;

		// Gaussian has been changed, update sum of the weights.
		weightSum = 0;
		for (const Gaussian& gauss : mixture)
			weightSum += gauss.weight;
	}

	// normalize the weights
	// sum of all weight is supposed to equal 1.
	float invWeightSum = 1.0 / weightSum;
	for (Gaussian& gauss : mixture)
		gauss.weight *= invWeightSum;

	// sort once again
	std::sort(std::begin(mixture), std::end(mixture), std::greater<Gaussian>());

	// estimate whether pixel belongs to foreground using probability equation given by Benedek & Sziranyi
	// calculate eplison for background
	const Gaussian& gauss = mixture[0];

	float epsilon_bg = 2 * log10(2 * M_PI);
	epsilon_bg += 3 * log10(sqrt(gauss.variance));
	epsilon_bg += 0.5 * (bgr[0] - gauss.meanB) * (bgr[0] - gauss.meanB) / gauss.variance;
	epsilon_bg += 0.5 * (bgr[1] - gauss.meanG) * (bgr[1] - gauss.meanG) / gauss.variance;
	epsilon_bg += 0.5 * (bgr[2] - gauss.meanR) * (bgr[2] - gauss.meanR) / gauss.variance;

	return epsilon_bg > foregroundThreshold;
}

int Background::findMixtureSize(const GaussianMixture& mixture) const
{
	int counter = 0;
	for (int i = 0; i < GAUSSIANS_PER_PIXEL; i++)
		counter += mixture[i].variance != 0 ? 1 : 0;
	return counter;
}

const Mat& Background::getCurrentBackground() const
{
	return currentBackground;
}

const Mat& Background::getCurrentStdDev() const
{
	return currentStdDev;
}
