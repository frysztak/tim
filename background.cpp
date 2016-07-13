#include "background.h"
#include "siltp.h"

Background::Background() :
	initialVariance(20),
	initialWeight(0.05),
	gaussiansPerPixel(3),
	learningRate(0.05),
	foregroundThreshold(8.5)
{
}

void Background::init(const Size& size)
{
	gaussians.reserve(size.area());
	for (int i = 0; i < size.area(); i++)
	{
		auto g = GaussianMixture();
		g.reserve(gaussiansPerPixel);
		gaussians.push_back(g);
	}

	currentBackground = Mat::zeros(size, CV_8UC3);
	currentTexture = Mat::zeros(Size(size.width - 4, size.height - 4), CV_32S);
}

void Background::processFrame(InputArray _src, OutputArray _foregroundMask)
{
	Mat src = _src.getMat(), foregroundMask = _foregroundMask.getMat();

	cvtColor(src, src, COLOR_BGR2Luv);

	for (int row = 0; row < src.rows; ++row)
	{
		uint8_t *foregroundMaskPtr = foregroundMask.ptr<uint8_t>(row);
		uint8_t *currentBackgroundPtr = currentBackground.ptr<uint8_t>(row);
		uint8_t *srcPtr = src.ptr<uint8_t>(row);
		int idx = src.cols * row; 

		for (int col = 0; col < src.cols; col++, idx++)
		{
			uint8_t x = *srcPtr++;
			uint8_t y = *srcPtr++;
			uint8_t z = *srcPtr++;
				
			GaussianMixture& mog = gaussians[idx];
			bool foreground = processPixel(Colour(x,y,z), mog);
			*foregroundMaskPtr++ = uint8_t(foreground ? 1 : 0);

			// update current background model (or rather, background image)
			auto& gauss = mog[0];
			*currentBackgroundPtr++ = gauss.miR;
			*currentBackgroundPtr++ = gauss.miG;
			*currentBackgroundPtr++ = gauss.miB;
		}
	}

	SILTP_16x2(currentBackground, currentTexture);
	medianBlur(foregroundMask, foregroundMask, 3);
}

bool Background::processPixel(const Colour& rgb, GaussianMixture& mixture)
{
	double weightSum = 0.0;
	bool matchFound = false;
	GaussianMixture matchedGaussianMixture;

	for (Gaussian& gauss : mixture)
	{
		float dR = gauss.miR - rgb.x;
		float dG = gauss.miG - rgb.y;
		float dB = gauss.miB - rgb.z;
		float distance = dR*dR + dG*dG + dB*dB;

		if (sqrt(distance) < 2.5*sqrt(gauss.variance) && !matchFound)
		{
			matchFound = true;

			float exponent = -0.5 / gauss.variance;
			exponent *= distance;
			
			float eta = pow(2 * M_PI, 3.0 / 2.0);
			eta *= pow(sqrt(gauss.variance), 3.0);
			eta = 1.0 / eta;
			eta *= exp(exponent);

			float rho = learningRate * eta;
			float oneMinusRho = 1.0 - rho;

			gauss.miR = oneMinusRho*gauss.miR + rho*rgb.x;
			gauss.miG = oneMinusRho*gauss.miG + rho*rgb.y;
			gauss.miB = oneMinusRho*gauss.miB + rho*rgb.z;
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
		if((int)mixture.size() < gaussiansPerPixel)
		{
			// add new Gaussian to the list
			mixture.emplace_back();
		}
		else 
		{
			// we can't add another Gaussian.
			// as per paper, let's modify least probable distribution.
			// but first, we have to sort by (weight/variance) parameter.
			std::sort(mixture.begin(), mixture.end(), std::greater<Gaussian>());
		}

		Gaussian& gauss = mixture.back();

		gauss.miR = rgb.x;	
		gauss.miG = rgb.y;
		gauss.miB = rgb.z;
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
	std::sort(mixture.begin(), mixture.end(), std::greater<Gaussian>());

	// estimate whether pixel belongs to foreground using probability equation given by Benedek & Sziranyi
	float L = (float)rgb.x;
	float u = (float)rgb.y;
	float v = (float)rgb.z;

	// calculate eplison for background
	const Gaussian& gauss = mixture[0];

	float epsilon_bg = 2 * log10(2 * M_PI);
	epsilon_bg += 3 * log10(sqrt(gauss.variance));
	epsilon_bg += 0.5 * (L - gauss.miR)*(L - gauss.miR) / gauss.variance;
	epsilon_bg += 0.5 * (u - gauss.miG)*(u - gauss.miG) / gauss.variance;
	epsilon_bg += 0.5 * (v - gauss.miB)*(v - gauss.miB) / gauss.variance;

	return epsilon_bg > foregroundThreshold;
}

const Mat& Background::getCurrentBackground() const
{
	return currentBackground;
}

const Mat& Background::getCurrentTexture() const
{
	return currentTexture;
}
