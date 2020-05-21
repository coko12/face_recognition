#include "personid.h"
#include <dirent.h>
#include <fstream>

bool get_filelist_from_dir1(std::string _path, std::vector<std::string>& _files)
{
	DIR* dir;	
	dir = opendir(_path.c_str());
	struct dirent* ptr;
	std::vector<std::string> file;
	while((ptr = readdir(dir)) != NULL)
	{
		if(ptr->d_name[0] == '.') {continue;}
		file.push_back(ptr->d_name);
	}
	closedir(dir);
	sort(file.begin(), file.end());
	_files = file;
}


Personid::Personid(const std::string &feature_path,const std::string name)
{
    this->name = name; 
    this->min_dst =0;
    std::vector<std::string> xml_files ;
    get_filelist_from_dir1(feature_path,xml_files);  
    if(xml_files.size()==0)
    {
        std::cout<<feature_path<<"is empty"<<std::endl;
        num_features = 0;
    } 
    else
    {
            
        for(auto xml_file:xml_files)
        {
            std::string xml_filedir = feature_path + "/" + xml_file;
            cv::FileStorage fsread(xml_filedir,cv::FileStorage::READ);
            cv::Mat feature_vec;
            fsread["feature"]>>feature_vec;
            feature_vecs.push_back(feature_vec);
        }
        num_features = feature_vecs.size();
    }
}

Personid::~Personid()  
{
}     

void Personid::calculMindistance(cv::Mat &v1)
{
    min_dst = 0;
    float dst = 0;
    for (auto feature_vec:feature_vecs)
    {
        dst = calculSimilar(v1,feature_vec);
        if(dst >min_dst)
        {
            min_dst = dst;
        }           
    }   
}

float Personid::get_dst()
{
   return min_dst;
}

std::string Personid::get_name()
{
    return this->name;
}

int Personid::get_numfeatures()
{
    return num_features;
}

float calculSimilar(cv::Mat& v1, cv::Mat& v2)
{
    double sum_square = 0.f;
    float* data1 = v1.ptr<float>(0);
    float* data2 = v2.ptr<float>(0);
    for (int i =0;i<512;i++)
    {
        double sub = data1[i] - data2[i];
        double square1 = sub * sub;
        sum_square = sum_square + square1;
    }
    return sum_square;
}