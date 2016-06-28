#ifndef STAUFFERGRIMSON_H
#define STAUFFERGRIMSON_H

#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>

using namespace cv;

static float constexpr initialWeight = 0.05f;
static float constexpr initialVariance = 20.0f;

struct Gaussian 
{
	float variance = initialVariance;
	float miR = 0;
	float miG = 0;
	float miB = 0;
	float weight = initialWeight;

	bool operator>(const Gaussian& other) const
	{
		return (this->weight/sqrt(this->variance)) > (other.weight/sqrt(other.variance));
	}
};

typedef Point3_<uint8_t> Colour;
typedef std::vector<Gaussian> GaussianMixture;

class StaufferGrimson
{
	friend class BenedekSziranyi;

	private:
		static int constexpr GaussiansPerPixel = 3;
		static float constexpr alpha = 0.05f;
		static float constexpr minimalWeightSum = 0.9f;

		static float constexpr oneMinusAlpha = 1.0f - alpha;			

		// matrix that holds Gaussians for each pixel
		std::vector<GaussianMixture> Gaussians;

		// current background model
		Mat Background;
		Mat ForegroundMask;

		bool SubstractPixel(const Colour& rgb, GaussianMixture& gaussians);

	public:
		StaufferGrimson();
		~StaufferGrimson();

		void Init(const Size& size);
		void Substract(InputArray src);
		void Dump(int idx);
};

#endif
