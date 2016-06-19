#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>

using namespace cv;

class StaufferGrimson
{
	private:
		static int constexpr GaussiansPerPixel = 3;
		static float constexpr alpha = 0.09f;
		static float constexpr initialWeight = 0.05f;
		static float constexpr initialVariance = 20.0f;
		static float constexpr minimalWeightSum = 0.8f;

		static float constexpr oneMinusAlpha = 1.0f - alpha;
			
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

		typedef std::vector<Gaussian> GaussianMixture;
		typedef Point3_<uint8_t> Pixel;

		// matrix that holds Gaussians for each pixel
		std::vector<GaussianMixture> Gaussians;

		// current background model
		Mat Background;

		bool SubstractPixel(const Pixel& rgb, GaussianMixture& gaussians);

	public:
		StaufferGrimson();
		~StaufferGrimson();

		void Init(const Size& size, Mat& foregroundMask);
		const Mat& Substract(InputArray src, OutputArray dst);
		void Dump(int idx);
};
