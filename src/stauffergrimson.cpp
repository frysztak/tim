#include <algorithm>
#include <cmath>
#include <functional>
#include "stauffergrimson.h"

StaufferGrimson::StaufferGrimson()
{
}

StaufferGrimson::~StaufferGrimson()
{
	this->Gaussians.clear();
}

void StaufferGrimson::Init(const Size &size, Mat& foregroundMask)
{
	auto num = size.area();
	this->Gaussians.reserve(num);
	for (int i = 0; i < num; i++)
	{
		auto g = GaussianMixture();
		g.reserve(GaussiansPerPixel);
		this->Gaussians.push_back(g);
	}

	this->Background = Mat(size, CV_8UC3);
	foregroundMask = Mat(size, CV_8UC3);
}

bool StaufferGrimson::SubstractPixel(const Pixel& rgb, GaussianMixture& mixture)
{
	double weightSum = 0.0;
	bool matchFound = false;
	GaussianMixture matchedGaussianMixture;

	for (Gaussian& gauss : mixture)
	{
		float dR = gauss.miR - rgb.z;
		float dG = gauss.miG - rgb.y;
		float dB = gauss.miB - rgb.x;
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

			float rho = alpha * eta;
			float oneMinusRho = 1.0 - rho;

			gauss.miR = oneMinusRho*gauss.miR + rho*rgb.z;
			gauss.miG = oneMinusRho*gauss.miG + rho*rgb.y;
			gauss.miB = oneMinusRho*gauss.miB + rho*rgb.x;
			gauss.variance = oneMinusRho*gauss.variance + rho*distance;
		}
		else
		{
			gauss.weight = oneMinusAlpha*gauss.weight;
		}

		weightSum += gauss.weight;
	}

	if(!matchFound)
	{
		// pixel didn't match any of currently existing Gaussians.
		if(mixture.size() < this->GaussiansPerPixel)
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

		gauss.miR = rgb.z;	
		gauss.miG = rgb.y;
		gauss.miB = rgb.x;
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

	// estimate background model (equation 9 in the paper)
	weightSum = 0;
	int B = 0;
	bool isForeground = false;

	for (const Gaussian& gauss : mixture)
	{
		if (weightSum > minimalWeightSum)
			break;

		B++;
		weightSum += gauss.weight;
	}

	for(int i = 0; i < B; i++)
	{
		const Gaussian& gauss = mixture[i];

		float dR = gauss.miR - rgb.z;
		float dG = gauss.miG - rgb.y;
		float dB = gauss.miB - rgb.x;
		float distance = sqrt(dR*dR + dG*dG + dB*dB);

		if (distance > 2.5*sqrt(gauss.variance))
		{
			isForeground = true;
			break;
		}
	}	

	return isForeground;
}

const Mat& StaufferGrimson::Substract(InputArray _src, OutputArray _dst)
{
	Mat src = _src.getMat();
	Mat dst = _dst.getMat();

	for(int r = 0; r < src.rows; r++)
	{
		for(int c = 0; c < src.cols; c++)
		{
			unsigned long idx = src.size().width*r + c;
			auto pixel = src.at<Pixel>(idx);
			
			GaussianMixture& gaussians = this->Gaussians[idx];
			bool foreground = this->SubstractPixel(pixel, gaussians);
			if(foreground)
				dst.at<Pixel>(idx) = Pixel(255, 255, 255);
			else
				dst.at<Pixel>(idx) = Pixel(0, 0, 0);

			// update current background model (or rather, background image)
			// most probable gaussian
			auto gauss = gaussians[0];
			this->Background.at<Pixel>(idx) = Pixel(gauss.miB, gauss.miG, gauss.miR);
		}
	}

	return this->Background;
}

void StaufferGrimson::Dump(int idx)
{
	GaussianMixture& gaussians = this->Gaussians[idx];
	for (const Gaussian& gauss : gaussians)
	{
		std::cout << "--- Gaussian" << std::endl;
		std::cout << "* weight: " << gauss.weight << std::endl;
		std::cout << "* variance: " << gauss.variance << std::endl;
		std::cout << "* mi{R,G,B}: " << gauss.miR << " " << gauss.miG << " " << gauss.miB << std::endl << std::endl;
	}
}
