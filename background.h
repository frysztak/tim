#ifndef BACKGROUND_H
#define BACKGROUND_H

#include <opencv2/opencv.hpp>

#define GAUSSIANS_PER_PIXEL 3

using namespace cv;

class Background
{
	public:
		struct Gaussian 
		{
			float meanB;
			float meanG;
			float meanR;
			float variance;
			float weight;
	
			bool operator>(const Gaussian& other) const
			{
				if (this->variance == 0 || other.variance == 0)
					return this->weight > other.weight;
				else
					return (this->weight/sqrt(this->variance)) > (other.weight/sqrt(other.variance));
			}

			friend std::ostream& operator<<(std::ostream& o, const Gaussian& g)
			{
				return o << "(B, G, R): (" << g.meanB << "," << g.meanG << "," << g.meanR << ")" << 
					"\t" << "(variance, weight): (" << g.variance << "," << g.weight << ")" ;
			}
		}; 
	
		typedef Gaussian GaussianMixture[GAUSSIANS_PER_PIXEL];
	
		Background();
		~Background();
		void init(const Size& size);
		void processFrame(InputArray _src, OutputArray _foregroundMask);
		void processFrameSIMD(InputArray _src, OutputArray _foregroundMask);
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

		bool processPixel(const Vec3b& rgb, GaussianMixture& mixture);
		uint32_t processPixelSSE2(const uint8_t* frame, float* gaussian, 
								  uint8_t* currentBackground, float* currentStdDev);

};

#endif
