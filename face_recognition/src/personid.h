#pragma once

#include <string>
#include <map>
#include "opencv2/opencv.hpp"


class Personid {
public:
	Personid(const std::string &feature_path,const std::string name);
    ~Personid();

    void calculMindistance(cv::Mat &v1);
    std::string get_name();
    float get_dst();
    int get_numfeatures();
private:
   std::string name;
   std::vector<cv::Mat> feature_vecs;
   float min_dst;
   int num_features;
};

float calculSimilar(cv::Mat& v1, cv::Mat& v2);