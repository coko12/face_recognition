#pragma once
#ifndef MOBILEFACENET_H_
#define MOBILEFACENET_H_
#include <string>
#include "net.h"
#include "opencv2/opencv.hpp"


class Recognize {
public:
	Recognize(const std::string &model_path);
	~Recognize();
	void start(const cv::Mat& img, cv::Mat &feature);
private:
	void RecogNet(ncnn::Mat& img_,ncnn::Mat& img_flip);
	ncnn::Net Recognet;
	ncnn::Mat feature_vec;
};



#endif // !MOBILEFACENET_H_