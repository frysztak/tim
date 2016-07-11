#include <algorithm>
#include "benedek.h"
#include "utils.h"

void BenedekSziranyi::Init(const Size& size)
{
	currentFrame = 1;

	this->FrameSize = size;
	bgs.Init(this->FrameSize);	

	InitShadowModel();
}

void BenedekSziranyi::ProcessFrame(InputArray _src, OutputArray _fg, OutputArray _sh)
{
	Mat inputFrame = _src.getMat();

	// update background model	
	// convert BGR to CIE L*u*v*
	cvtColor(inputFrame, inputFrame, COLOR_BGR2Luv);
	bgs.Substract(inputFrame);

	// TODO: update microstructure model
	
	DetectForeground(inputFrame);
//	if (currentFrame % shadowModelUpdateRate == 0)
//		UpdateShadowModel();

	this->ForegroundMask.copyTo(_fg);
	this->ShadowMask.copyTo(_sh);

	currentFrame++;
}

void BenedekSziranyi::DetectForeground(InputArray _src)
{
	Mat inputFrame = _src.getMat();
	this->ForegroundMask = Mat::zeros(FrameSize, CV_8U);
	this->ShadowMask = Mat::zeros(FrameSize, CV_8U);

	// preliminary detection
	for(int idx = 0; idx < FrameSize.area(); idx++)
	{
		auto& colour = inputFrame.at<Colour>(idx);
		float L = (float)colour.x;
		float u = (float)colour.y;
		float v = (float)colour.z;

		// calculate eplison for background
		const Gaussian& gauss = bgs.Gaussians[idx][0];

		float epsilon_bg = 2 * log10(2 * M_PI);
		epsilon_bg += 3 * log10(sqrt(gauss.variance));
		epsilon_bg += 0.5 * pow(L - gauss.miR, 2) / gauss.variance;
		epsilon_bg += 0.5 * pow(u - gauss.miG, 2) / gauss.variance;
		epsilon_bg += 0.5 * pow(v - gauss.miB, 2) / gauss.variance;


		if (shadowDetectionEnabled)
		{	
			float epsilon_sh = 2 * log10(2 * M_PI);
			epsilon_sh += log10(sqrt(shadowModel.L_variance));
			epsilon_sh += log10(sqrt(shadowModel.u_variance));
			epsilon_sh += log10(sqrt(shadowModel.v_variance));
			epsilon_sh += 0.5 * pow(L - shadowModel.L_mean, 2) / shadowModel.L_variance;
			epsilon_sh += 0.5 * pow(u - shadowModel.u_mean, 2) / shadowModel.u_variance;
			epsilon_sh += 0.5 * pow(v - shadowModel.v_mean, 2) / shadowModel.v_variance;

			bool addedToQ = false;	
			if (epsilon_sh < foregroundThreshold)
			{
				// this pixel is a shadow
				ShadowMask.at<uint8_t>(idx) = 1;

				shadowModel.Wu_t.push_back(u - gauss.miG);
				shadowModel.Q.emplace_back(L, currentFrame);
				addedToQ = true;
			}
			else if (epsilon_bg > foregroundThreshold && epsilon_sh > foregroundThreshold)
			{
				// it's a foreground
				ForegroundMask.at<uint8_t>(idx) = 1;
				if(!addedToQ)
					shadowModel.Q.emplace_back(L, currentFrame);
			}
		}
		else
		{
			if (epsilon_bg > foregroundThreshold)
			{
				// it's a foreground
				ForegroundMask.at<uint8_t>(idx) = 1;
				bgs.BackgroundProbability.at<float>(idx) = epsilon_bg;
			//	shadowModel.Q.emplace_back(L, currentFrame);
			}
			else
			{
				bgs.BackgroundProbability.at<float>(idx) = 0;
			}
		}
	}

	// update shadow model (L_mean)
	//if (shadowModel.Q.size() >= Qmin)
	//{
	//	while (shadowModel.Q.size() > Qmax)
	//	{
	//		// find eldest timestamp
	//		auto eldest = *std::min_element(shadowModel.Q.cbegin(), shadowModel.Q.cend(), 
	//				[](const auto& A, const auto& B) { return std::get<1>(A) < std::get<1>(B); });
	//		uint32_t timestamp = std::get<1>(eldest);

	//		auto endIt = shadowModel.Q.begin() + 1000;
	//		shadowModel.Q.erase(std::remove_if(shadowModel.Q.begin(), endIt, 
	//				[=](const auto& A) { return std::get<1>(A) == timestamp; }), endIt);
	//	}
	//}
	//shadowModel.L_mean = argmax(shadowModel.Q);
	
	// second run, using moving window
	if (!windowPassEnabled)
		return;

	Mat Vs, Fs_mask, FsD_mask;
	for(int r = 0; r < ForegroundMask.rows; r++)
	{
		for(int c = 0; c < ForegroundMask.cols; c++)
		{
			if (ForegroundMask.at<uint8_t>(r, c) == 0)
				continue;

			// calculate window coordinates
			int startX, endX, startY, endY;
			// ideally, current pixel is in the middle of the window.
			// but that's not always possible.
			startX = c - windowSize / 2;
			if (startX < 0) startX = 0;
			endX = startX + windowSize;
			if (endX > inputFrame.cols) endX = inputFrame.cols;

			startY = r - windowSize / 2;
			if (startY < 0) startY = 0;
			endY = startY + windowSize;
			if (endY > inputFrame.rows) endY = inputFrame.rows;

			// in order to get Fs, we'll need Vs and a corresponding piece of foreground mask
			Rect rect = Rect(startX, startY, endX - startX, endY - startY);
	   		Vs = inputFrame(rect);
			Fs_mask = ForegroundMask(rect);
			FsD_mask = Mat::zeros(Vs.size(), CV_8U);

			// instead of constructing separate matrix Fs^D, we'll construct a mask
			// and pass it to meanStdDev function.
			auto colour = inputFrame.at<Colour>(r, c);
			float L = (float)colour.x;
			float u = (float)colour.y;
			float v = (float)colour.z;

			for (int idx = 0; idx < Fs_mask.size().area(); idx++)
			{
				if (Fs_mask.at<uint8_t>(idx) == 0)
					continue;

				auto colour_ = Vs.at<Colour>(idx);
				float L_ = (float)colour_.x;
				float u_ = (float)colour_.y;
				float v_ = (float)colour_.z;

				float dL = L - L_;
				float du = u - u_;
				float dv = v - v_;
				float distance = sqrt(dL*dL + du*du + dv*dv);

				if (distance < tau)
					FsD_mask.at<uint8_t>(idx) = 1;
			}

			Scalar mean, stdDev;
			Mat mask;
		    bitwise_and(Fs_mask, FsD_mask, mask);
			meanStdDev(Vs, mean, stdDev, mask);

			int Fs_num = countNonZero(Fs_mask);
			float Kappa_s1 = float(countNonZero(FsD_mask))/float(Fs_num);
			float Kappa_s2 = float(Fs_num)/float((endX - startX)*(endY - startY));

			float temp = 1 + exp(-(Kappa_s2 - kappa_min/2.0));
			float Kappa_s = Kappa_s1 / temp;

			float epsilon_fg = -log10(Kappa_s);

			// now we ought to calculate eta
			float dL = mean[0] - L;
			float du = mean[1] - u;
			float dv = mean[2] - v;

			float exponent = -0.5;
			exponent *= pow(dL/stdDev[0], 2) + pow(du/stdDev[1], 2) + pow(dv/stdDev[2], 2);
		
			float eta = pow(2 * M_PI, 3.0 / 2.0);
			eta *= pow(stdDev[0] * stdDev[1] * stdDev[2], 3.0 / 2.0);
			eta = 1.0 / eta;
			eta *= exp(exponent);
			
			epsilon_fg -= log10(eta);

			ForegroundMask.at<uint8_t>(r, c) = epsilon_fg > foregroundThreshold2 ? 1 : 0;
		}
	}
}

void BenedekSziranyi::InitShadowModel()
{
	Mat shadowMask = imread("/mnt/things/tim/masks/act_shadows_mask.png", 0);
	Mat shadow = imread("/mnt/things/tim/masks/act_shadows.png");

	cvtColor(shadow, shadow, COLOR_BGR2Luv);

	Scalar mean, stdDev;
	meanStdDev(shadow, mean, stdDev, shadowMask);

	shadowModel.L_mean = mean[0];
	shadowModel.u_mean = mean[1];
	shadowModel.v_mean = mean[2]; 
	shadowModel.L_variance = stdDev[0] * stdDev[0];
	shadowModel.u_variance = stdDev[1] * stdDev[1];
	shadowModel.v_variance = stdDev[2] * stdDev[2];
}

void BenedekSziranyi::UpdateShadowModel()
{
	Mat wu(shadowModel.Wu_t, false);

	Scalar mean, stdDev;
	meanStdDev(wu, mean, stdDev);

	float xi = shadowModel.Wu_t.size() / (float(FrameSize.area()) * shadowModelUpdateRate * 150);

	shadowModel.u_mean = (1.0 - xi)*shadowModel.u_mean + xi*mean[0];
	shadowModel.v_mean = (1.0 - xi)*shadowModel.v_mean + xi*mean[0];
	shadowModel.u_variance = (1.0 - xi)*shadowModel.u_variance + xi*stdDev[0]*stdDev[0];
	shadowModel.v_variance = (1.0 - xi)*shadowModel.v_variance + xi*stdDev[0]*stdDev[0];

	std::cout << "\tUpdating shadow model..." << std::endl;
	std::cout << "#Wu = " << shadowModel.Wu_t.size() << std::endl;
	std::cout << "xi = " << xi << std::endl;
	std::cout << "u_mean = " << shadowModel.u_mean << ", v_mean = " << shadowModel.v_mean << std::endl;
	std::cout << "u_variance = " << shadowModel.u_variance << ", v_variance = " << shadowModel.v_variance << std::endl;
	std::cout << "L_mean = " << shadowModel.L_mean << ", L_variance = " << shadowModel.L_variance << std::endl;

	shadowModel.Wu_t.clear();
}

void BenedekSziranyi::ToggleShadowDetection()
{
	shadowDetectionEnabled = !shadowDetectionEnabled;
}

const Mat& BenedekSziranyi::GetStaufferBackgroundModel()
{
	return bgs.Background;
}

const Mat& BenedekSziranyi::GetStaufferForegroundMask()
{
	return bgs.ForegroundMask;
}
