#include <algorithm>
#include "benedek.h"
#include "utils.h"

void BenedekSziranyi::Init(const Size& size)
{
	currentFrame = 1;

	this->FrameSize = size;
	bgs.Init(this->FrameSize);

	this->Models.reserve(this->FrameSize.area());
	for (int i = 0; i < this->FrameSize.area(); i++)
	{
		auto p = Pixel();
		this->Models.push_back(p);
	}

	InitShadowModel();
}

void BenedekSziranyi::ProcessFrame(InputArray _src, OutputArray _fg, OutputArray _sh)
{
	Mat inputFrame = _src.getMat();

	// update background model
	cvtColor(inputFrame, inputFrame, COLOR_BGR2RGB);
	bgs.Substract(inputFrame);

	// convert BGR to CIE L*u*v*
	cvtColor(inputFrame, inputFrame, COLOR_RGB2Luv);
	cvtColor(bgs.Background, bgs.Background, COLOR_RGB2Luv);

	// TODO: update microstructure model
	
	DetectForeground(inputFrame);
	if (currentFrame % ShadowModelUpdateRate == 0)
		UpdateShadowModel();

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
	for(int r = 0; r < inputFrame.rows; r++)
	{
		for(int c = 0; c < inputFrame.cols; c++)
		{
			unsigned long idx = inputFrame.size().width*r + c;
			auto& colour = inputFrame.at<Colour>(idx);
			float L = (float)colour.x;
			float u = (float)colour.y;
			float v = (float)colour.z;

			// calculate eplison for background
			const Gaussian& gauss = bgs.Gaussians[idx][0];
			Colour& background = bgs.Background.at<Colour>(idx);

			float epsilon_bg = 2 * log10(2 * M_PI);
			epsilon_bg += 3 * log10(sqrt(gauss.variance));
			epsilon_bg += 0.5 * pow(L - background.x, 2) / gauss.variance;
			epsilon_bg += 0.5 * pow(u - background.y, 2) / gauss.variance;
			epsilon_bg += 0.5 * pow(v - background.z, 2) / gauss.variance;
		
			float epsilon_sh = 2 * log10(2 * M_PI);
			epsilon_sh += log10(sqrt(shadowModel.L_variance));
			epsilon_sh += log10(sqrt(shadowModel.u_variance));
			epsilon_sh += log10(sqrt(shadowModel.v_variance));
			epsilon_sh += 0.5 * pow(L - shadowModel.L_mean, 2) / shadowModel.L_variance;
			epsilon_sh += 0.5 * pow(u - shadowModel.u_mean, 2) / shadowModel.u_variance;
			epsilon_sh += 0.5 * pow(v - shadowModel.v_mean, 2) / shadowModel.v_variance;

			bool addedToQ = false;	
			if (epsilon_sh < ForegroundThreshold)
			{
				// this pixel is a shadow
				ShadowMask.at<uint8_t>(idx) = 1;

				shadowModel.Wu_t.push_back(u - background.y);
				shadowModel.Q.emplace_back(L, currentFrame);
				addedToQ = true;
			}
			else if (epsilon_bg > ForegroundThreshold && epsilon_sh > ForegroundThreshold)
			{
				// it's a foreground
				ForegroundMask.at<uint8_t>(idx) = 1;
				if(!addedToQ)
					shadowModel.Q.emplace_back(L, currentFrame);
			}
		}
	}

	// update shadow model (L_mean)
	if (shadowModel.Q.size() >= Qmin)
	{
		while (shadowModel.Q.size() > Qmax)
		{
			// find eldest timestamp
			auto eldest = *std::min_element(shadowModel.Q.cbegin(), shadowModel.Q.cend(), 
					[](const auto& A, const auto& B) { return std::get<1>(A) < std::get<1>(B); });
			uint32_t timestamp = std::get<1>(eldest);

			auto endIt = shadowModel.Q.begin() + 1000;
			shadowModel.Q.erase(std::remove_if(shadowModel.Q.begin(), endIt, 
					[=](const auto& A) { return std::get<1>(A) == timestamp; }), endIt);
		}
	}
	shadowModel.L_mean = argmax(shadowModel.Q);
	
	// second run, using moving window
	if (!FOREGROUND_SECOND_PASS)
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
			startX = c - WindowSize / 2;
			if (startX < 0) startX = 0;
			endX = startX + WindowSize;
			if (endX > inputFrame.cols) endX = inputFrame.cols;

			startY = r - WindowSize / 2;
			if (startY < 0) startY = 0;
			endY = startY + WindowSize;
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

			for (int ri = 0; ri < Fs_mask.rows; ri++)
			{
				for (int ci = 0; ci < Fs_mask.cols; ci++)
				{
					if (Fs_mask.at<uint8_t>(ri, ci) == 0)
						continue;

					auto colour_ = Vs.at<Colour>(ri, ci);
					float L_ = (float)colour_.x;
					float u_ = (float)colour_.y;
					float v_ = (float)colour_.z;

					float dL = L - L_;
					float du = u - u_;
					float dv = v - v_;
					float distance = sqrt(dL*dL + du*du + dv*dv);

					if (distance < Tau)
						FsD_mask.at<uint8_t>(ri, ci) = 1;
				}
			}

			Scalar mean, stdDev;
			Mat mask;
		    bitwise_and(Fs_mask, FsD_mask, mask);
			meanStdDev(Vs, mean, stdDev, mask);

			int Fs_num = countNonZero(Fs_mask);
			float Kappa_s1 = float(countNonZero(FsD_mask))/float(Fs_num);
			float Kappa_s2 = float(Fs_num)/float((endX - startX)*(endY - startY));

			float temp = 1 + exp(-(Kappa_s2 - Kappa_min/2.0));
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

			ForegroundMask.at<uint8_t>(r, c) = epsilon_fg > ForegroundThreshold2 ? 1 : 0;
		}
	}
}

void BenedekSziranyi::InitShadowModel()
{
	Mat shadowMask = imread("/mnt/things/car detection/videos/act_shadows_mask.png", 0);
	Mat shadow = imread("/mnt/things/car detection/videos/act_shadows.png");

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
	std::cout << "\tUpdating shadow model..." << std::endl;
	Mat wu(shadowModel.Wu_t, false);

	Scalar mean, stdDev;
	meanStdDev(wu, mean, stdDev);

	std::cout << "#Wu = " << shadowModel.Wu_t.size() << std::endl;

	float xi = shadowModel.Wu_t.size() / (float(FrameSize.area()) * ShadowModelUpdateRate * 150);
	std::cout << "xi = " << xi << std::endl;

	shadowModel.u_mean = (1.0 - xi)*shadowModel.u_mean + xi*mean[0];
	shadowModel.v_mean = (1.0 - xi)*shadowModel.v_mean + xi*mean[0];
	shadowModel.u_variance = (1.0 - xi)*shadowModel.u_variance + xi*stdDev[0]*stdDev[0];
	shadowModel.v_variance = (1.0 - xi)*shadowModel.v_variance + xi*stdDev[0]*stdDev[0];

	std::cout << "u_mean = " << shadowModel.u_mean << ", v_mean = " << shadowModel.v_mean << std::endl;
	std::cout << "u_variance = " << shadowModel.u_variance << ", v_variance = " << shadowModel.v_variance << std::endl;

	shadowModel.Wu_t.clear();
}

const Mat& BenedekSziranyi::GetStaufferBackgroundModel()
{
	return bgs.Background;
}

const Mat& BenedekSziranyi::GetStaufferForegroundMask()
{
	return bgs.ForegroundMask;
}
