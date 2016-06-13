#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>

using namespace cv;

class StaufferGrimson
{
	private:
		static const int GaussiansPerPixel = 4;
		static float constexpr alpha = 0.002f;
		static float constexpr oneMinusAlpha = 1.0f - alpha;
		static float constexpr rho = 0.0005f;
		static float constexpr oneMinusRho = 1.0f - rho;
			
		struct Gaussian 
		{
			float variance = 16.0f;
			float miR = 0;
			float miG = 0;
			float miB = 0;
			float weight = 0.1f;

			bool operator>(const Gaussian& other) const
			{
				return (this->weight/sqrt(this->variance)) > (other.weight/sqrt(other.variance));
			}
		};

		typedef std::vector<Gaussian> GaussianMixture;
		typedef Point3_<uint8_t> Pixel;

		// matrix that holds Gaussians for each pixel
		std::vector<GaussianMixture> Gaussians;

		bool SubstractPixel(const Pixel& rgb, GaussianMixture& gaussians);

	public:
		StaufferGrimson();
		~StaufferGrimson();

		void Init(const Size& size);
		void Substract(InputArray src, OutputArray dst);
		void Dump(int idx);
};
