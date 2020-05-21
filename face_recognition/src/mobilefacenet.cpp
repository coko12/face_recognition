/*
created by L. 2018.05.16
*/

#include "mobilefacenet.h"

Recognize::Recognize(const std::string &model_path) {
    std::string param_files = model_path + "/IR_50.param";
    std::string bin_files = model_path + "/IR_50.bin";
    Recognet.load_param(param_files.c_str());
    Recognet.load_model(bin_files.c_str());
    feature_vec.create(512,4);
}

Recognize::~Recognize() {
    Recognet.clear();
    feature_vec.release();
}

void Recognize::RecogNet(ncnn::Mat& img_,ncnn::Mat& img_flip) {
    ncnn::Extractor ex = Recognet.create_extractor();
    //ex.set_num_threads(2);
    ex.set_light_mode(true);
    ex.input("input.1", img_);
    ncnn::Mat out;
    ex.extract("561", out);

    //新定义一个执行器运行镜像图像，每运行一次最后都重新定义执行器。
    //经测试如果不新定义则out1输出与 out相同
    ncnn::Extractor ex1 = Recognet.create_extractor();
    ex1.input("input.1", img_flip);
    ncnn::Mat out1;
    ex1.extract("561", out1);    
    for (int j = 0; j < 512; j++)
    {
        feature_vec[j] = out[j] + out1[j];
    }
    float v1_sum = 0.f;
    for (int j = 0; j < 512; j++)
    {
      v1_sum += feature_vec[j] * feature_vec[j];
    }
    float v1_sqr = sqrt(v1_sum); 
    for (int j = 0; j < 512; j++)
    {
      float a = feature_vec[j];
      feature_vec[j] = feature_vec[j] / v1_sqr;
    }       
}

void Recognize::start(const cv::Mat& img, cv::Mat &feature) {
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, 112, 112);    
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float std_vals[3]  = {0.0078125f, 0.0078125f, 0.0078125f};
    ncnn_img.substract_mean_normalize(mean_vals, std_vals);

    cv::Mat imgflip;
    cv::flip(img,imgflip,1);
    ncnn::Mat ncnn_imgflip = ncnn::Mat::from_pixels_resize(imgflip.data, ncnn::Mat::PIXEL_BGR2RGB, imgflip.cols, imgflip.rows, 112, 112);
    ncnn_imgflip.substract_mean_normalize(mean_vals, std_vals);  
    RecogNet(ncnn_img,ncnn_imgflip);
    if(feature.empty())
        feature.create(1,this->feature_vec.w , CV_32F);
    float* data = feature.ptr<float>(0);
    for (int j = 0; j < this->feature_vec.w; j++)
    {
        data[j] =this->feature_vec[j];
    }
}


