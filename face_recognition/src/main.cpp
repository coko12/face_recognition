#include "mtcnn.h"
#include "mobilefacenet.h"
#include "personid.h"
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <fstream>
#include <unistd.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>


using namespace cv;
#define MAXFACEOPEN 1

struct match_result{
    std::string name;
    float dst;
};


bool get_filelist_from_dir(std::string _path, std::vector<std::string>& _files)
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

void split(const string& src, const string& separator, vector<string>& dest)
{
    string str = src;
    string substring;
    string::size_type start = 0, index;

    do
    {
        index = str.find_first_of(separator,start);
        if (index != string::npos)
        {    
            substring = str.substr(start,index-start);
            dest.push_back(substring);
            start = str.find_first_not_of(separator,index);
            if (start == string::npos) return;
        }
    }while(index != string::npos);
    //the last token
    substring = str.substr(start);
    dest.push_back(substring);
} 

cv::Mat getsrc_roi(std::vector<cv::Point2f> x0, std::vector<cv::Point2f> dst)
{
    int size = dst.size();
    cv::Mat A = cv::Mat::zeros(size * 2, 4, CV_32FC1);
    cv::Mat B = cv::Mat::zeros(size * 2, 1, CV_32FC1);

    for (int i = 0; i < size; i++) {
        A.at<float>(i << 1, 0) = x0[i].x;// roi_dst[i].x;
        A.at<float>(i << 1, 1) = -x0[i].y;
        A.at<float>(i << 1, 2) = 1;
        A.at<float>(i << 1, 3) = 0;
        A.at<float>(i << 1 | 1, 0) = x0[i].y;
        A.at<float>(i << 1 | 1, 1) = x0[i].x;
        A.at<float>(i << 1 | 1, 2) = 0;
        A.at<float>(i << 1 | 1, 3) = 1;

        B.at<float>(i << 1) = dst[i].x;
        B.at<float>(i << 1 | 1) = dst[i].y;
    }

    cv::Mat roi = cv::Mat::zeros(2, 3, A.type());
    cv::Mat AT = A.t();
    cv::Mat ATA = A.t() * A;
    cv::Mat R = ATA.inv() * AT * B;

    roi.at<float>(0, 0) = R.at<float>(0, 0);
    roi.at<float>(0, 1) = -R.at<float>(1, 0);
    roi.at<float>(0, 2) = R.at<float>(2, 0);
    roi.at<float>(1, 0) = R.at<float>(1, 0);
    roi.at<float>(1, 1) = R.at<float>(0, 0);
    roi.at<float>(1, 2) = R.at<float>(3, 0);
    return roi;
}

void align(cv::Mat image, Bbox &bbox,Mat &face)
{
    double dst_landmark[10] = {
                38.2946, 73.5318, 55.0252, 41.5493, 70.7299,
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041 };

    vector<cv::Point2f> coord5points;
    vector<cv::Point2f> facePointsByMtcnn;

    //face.rect[0] = bboxes[i].x1;
    //face.rect[1] = bboxes[i].y1;
    //face.rect[2] = bboxes[i].x2 - bboxes[i].x1 + 1;
    //face.rect[3] = bboxes[i].y2 - bboxes[i].y1 + 1;

    for (int j = 0; j < 5; j ++) {
        facePointsByMtcnn.push_back(cvPoint(bbox.ppoint[j], bbox.ppoint[j + 5]));
        coord5points.push_back(cv::Point2f(dst_landmark[j], dst_landmark[j + 5]));
    }

    cv::Mat warp_mat = estimateRigidTransform(facePointsByMtcnn, coord5points, false);
    if (warp_mat.empty()) {
        warp_mat = getsrc_roi(facePointsByMtcnn, coord5points);
    }
    warp_mat.convertTo(warp_mat, CV_32FC1);
    //face = cv::Mat::zeros(112, 112, image.type());
    warpAffine(image, face, warp_mat, face.size());
}

int gallery_face_aglin(std::string &face_path,std::string &aglin_path,MTCNN &mtcnn)
{
    if (access(aglin_path.c_str(), 0) == -1)
    {
        std::cout<<aglin_path<<"is not exist."<<std::endl;
        return -1;
    }
    std::vector<std::string> imgfiles ;
    get_filelist_from_dir(face_path,imgfiles);
    if(imgfiles.size()==0)
    {
        std::cout<<face_path<<"has no face image"<<std::endl;
        return 0;
    }
    else
    {
        int num = 0;
        for (auto imgfile:imgfiles)
        {
            int flag =1;
            string::size_type idx;
            string::size_type idx1;        
            idx = imgfile.find("jpg");
            idx1 = imgfile.find("png");
            if(idx1 == string::npos and idx == string::npos)
                flag = 0;
            if(flag ==0)
            {
                std::cout<<imgfile<<"is not image format"<<std::endl;
                continue;
            }  
            string imgdir = face_path + string("/") + imgfile; 
            Mat img = imread(imgdir);
            if (img.empty())  // 判断读入的图片是否为空
            {
                std::cout<<"Could not load "<<imgdir<<std::endl;
                continue;
            }
            ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows);
            std::vector<Bbox> finalBbox;            
            #if(MAXFACEOPEN==1)
                mtcnn.detectMaxFace(ncnn_img, finalBbox);
            #else
                mtcnn.detect(ncnn_img, finalBbox);
            #endif
            const int num_box = finalBbox.size();
            std::vector<cv::Rect> bbox;
            //bbox.resize(num_box);

            if (num_box == 1) {
                //cv::Mat ROI(img, Rect(finalBbox[0].x1, finalBbox[0].y1, finalBbox[0].x2 - finalBbox[0].x1 + 1, finalBbox[0].y2 - finalBbox[0].y1 + 1));
                //cv::Mat croppedImage;
                //ROI.copyTo(croppedImage);

                Mat croppedImage = cv::Mat::zeros(112, 112, img.type());
                align(img,finalBbox[0],croppedImage);

                string aglindir = aglin_path + string("/") + imgfile;
                imwrite(aglindir,croppedImage);
                num +=1;
            } 
            else 
            {
                std::cout << "no face detected or too much faces" << std::endl;
            }                
        }
        return num;
    }
}

int get_gallery_feature(std::string &aglin_path,std::string &feature_path,Recognize &recognize)
{
    if (access(feature_path.c_str(), 0) == -1)
    {
        std::cout<<feature_path<<"is not exist."<<std::endl;
        return -1;
    }  

    std::vector<std::string> imgfiles ;
    get_filelist_from_dir(aglin_path,imgfiles);
    if(imgfiles.size()==0)
    {
        std::cout<<aglin_path<<"has no face image"<<std::endl;
        return 0;
    }
    else
    {
        int num = 0;
        for (auto imgfile:imgfiles)
        {
            int flag =1;
            string::size_type idx;
            string::size_type idx1;        
            idx = imgfile.find("jpg");
            idx1 = imgfile.find("png");
            if(idx1 == string::npos and idx == string::npos)          
                flag = 0;
            if(flag ==0)
            {
                std::cout<<imgfile<<"is not image format"<<std::endl;
                continue;
            }  
            string imgdir = aglin_path + string("/") + imgfile; 
            Mat img = imread(imgdir,CV_LOAD_IMAGE_COLOR);
            if (img.empty())  // 判断读入的图片是否为空
            {
                std::cout<<"Could not load "<<imgdir<<std::endl;
                continue;
            }
            vector<string> d;
            split(imgfile,".",d); 
            string xmldir = feature_path + string("/") + d[0];
            if(d.size()>1)
            {
                for(int i =1;i<d.size()-1;i++)
                {
                    xmldir = xmldir + d[i];
                }
            }
            xmldir = xmldir + ".xml";
            cv::Mat feature_vec;
            recognize.start(img,feature_vec);
	        cv::FileStorage fswrite(xmldir, FileStorage::WRITE);
	        fswrite << "feature" << feature_vec;
            fswrite.release();                        
            num +=1;
        }
        return num;
    }
}

int add_gallery(std::string &gallery,std::string &personname,Recognize &recognize,MTCNN &mtcnn)
{
    string gallery_original = "face_image";
    string gallery_aglin = "face_aglin";
    string gallery_xml = "face_feature";
    string face_path = gallery + "/" + gallery_original + "/" + personname;
    string aglin_path = gallery + "/" + gallery_aglin + "/"+ personname;
    string xml_path = gallery + "/" + gallery_xml + "/"+ personname;
    if (access(face_path.c_str(), 0) == -1)
    {
        std::cout<<face_path<<"is not exist"<<std::endl;//"is not exist."；
        return -1;
    }
    if (access(aglin_path.c_str(), 0) == -1)
    {
        cout<<aglin_path<<" is not existing"<<endl;
		cout<<"now make it"<<endl;
        int flag=mkdir(aglin_path.c_str(), 0777); 
		if (flag == 0)
        {
            cout<<aglin_path<<"make successfully"<<endl;
		} 
        else 
        {
            cout<<aglin_path<<"make errorly"<<endl;
            return -1;
		}           
    }
    if (access(xml_path.c_str(), 0) == -1)
    {
        cout<<xml_path<<" is not existing"<<endl;
		cout<<"now make it"<<endl;
        int flag=mkdir(xml_path.c_str(), 0777); 
		if (flag == 0)
        {
            cout<<xml_path<<"make successfully"<<endl;
		} 
        else 
        {
            cout<<xml_path<<"make errorly"<<endl;
            return -1;
		}           
    } 
    int aglin_flag = gallery_face_aglin(face_path,aglin_path,mtcnn);
    int xml_flag =0;
    //int aglin_flag = 1;
    if (aglin_flag>0)
    {
       xml_flag = get_gallery_feature(aglin_path,xml_path,recognize);
    } 
    return  xml_flag;
}

int init_gallery(std::string &gallery,Recognize &recognize,MTCNN &mtcnn)
{
    // const char *model_path = "../models";
    // Recognize recognize(model_path); 
    // MTCNN mtcnn(model_path);         
    string gallery_original =gallery + "/" + "face_image";
    string gallery_aglin = gallery + "/" + "face_aglin";
    string gallery_xml = gallery + "/" + "face_feature";
    if (access(gallery_original.c_str(), 0) == -1)
    {
        std::cout<<gallery_original<<" is not existing"<<endl;
        return -1;
    }
    if (access(gallery_aglin.c_str(), 0) == -1)
    {
        cout<<gallery_aglin<<" is not existing"<<endl;
		cout<<"now make it"<<endl;
        int flag=mkdir(gallery_aglin.c_str(), 0777); 
		if (flag == 0)
        {
            cout<<gallery_aglin<<"make successfully"<<endl;
		} 
        else 
        {
            cout<<gallery_aglin<<"make errorly"<<endl;
            return -1;
		}           
    } 
     if (access(gallery_xml.c_str(), 0) == -1)
    {
        cout<<gallery_xml<<" is not existing"<<endl;
		cout<<"now make it"<<endl;
        int flag=mkdir(gallery_xml.c_str(), 0777); 
		if (flag == 0)
        {
            cout<<gallery_xml<<"make successfully"<<endl;
		} 
        else 
        {
            cout<<gallery_xml<<"make errorly"<<endl;
            return -1;
		}           
    } 
    std::vector<std::string> imgfiles ;
    get_filelist_from_dir(gallery_original,imgfiles);
    int num = 0;
    if(imgfiles.size()==0)
    {
        std::cout<<gallery_original<<"is empty"<<std::endl;
    }
    else
    {   
        for(auto person:imgfiles) 
        {
            int xmlflag = add_gallery(gallery,person,recognize,mtcnn);
            if(xmlflag>0)
                num += xmlflag;
        }
    }
    return num;
}

int init_face_library(std::string &gallery,vector<Personid> &face_library)
{
    string gallery_xml = gallery + "/" + "face_feature";
    if (access(gallery_xml.c_str(), 0) == -1)
    {
        std::cout<<gallery_xml<<" is not existing"<<endl;
        return -1;
    } 

    std::vector<std::string> persons ;
    get_filelist_from_dir(gallery_xml,persons);
    int num = 0;
    if(persons.size()==0)
    {
        std::cout<<gallery_xml<<"is empty"<<std::endl;
        return -1;
    }
    for (auto person:persons)
    {
        string xml_path = gallery_xml + "/" + person;

        Personid personid(xml_path,person);
        if(personid.get_numfeatures() > 0)
        {
            face_library.push_back(personid);
            num +=1;
        }                   
    }
    return num;
}

void faceimg_verification(Mat &faceimg,vector<Personid> &face_library,MTCNN &mtcnn,Recognize &recognize,std::vector<match_result> &results,float threshold)
{
 
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(faceimg.data, ncnn::Mat::PIXEL_BGR2RGB, faceimg.cols, faceimg.rows);
    std::vector<Bbox> finalBbox;            
    #if(MAXFACEOPEN==1)
        mtcnn.detectMaxFace(ncnn_img, finalBbox);
    #else
        mtcnn.detect(ncnn_img, finalBbox);
    #endif
    const int num_box = finalBbox.size();
    std::vector<cv::Rect> bbox;
    bbox.resize(num_box);

    if (num_box == 1) {
        cv::Mat ROI(faceimg, Rect(finalBbox[0].x1, finalBbox[0].y1, finalBbox[0].x2 - finalBbox[0].x1 + 1, finalBbox[0].y2 - finalBbox[0].y1 + 1));
        cv::Mat croppedImage;
        ROI.copyTo(croppedImage);
        cv::Mat feature_vec;
        recognize.start(croppedImage,feature_vec);        
        for (auto personid:face_library) 
        {
            personid.calculMindistance(feature_vec);
            float dst = personid.get_dst();
            std::string name = personid.get_name();
            if(dst<threshold)
            {
                match_result result;
                result.dst = dst;
                result.name =name;
                results.push_back(result);
            }
        }
        }        
}


void lfw_test(string pair,string gallery)//,match_result)
{
    ifstream infile;
    infile.open(pair.data());//将文件流对象与文件连接起来
    assert(infile.is_open());//若失败,则输出错误消息,并终止程序运行
    string s;
    float num =0.f;
    float num_face =0.f;
    float num_mtcnn=0.f;
    while(getline(infile,s))
    {
        //cout<<s<<endl;
        vector<string> d;
        split(s,"\t",d); 
        if(d.size()==3)
        {
            std::string xml_file = gallery + d[0] + "/" + d[0] + "_000" + d[1] + ".xml";
            std::string xml_file1 = gallery + d[0] + "/"+d[0] +  "_000" + d[2] + ".xml";        
            if(d[1].size()==2)
            {
                xml_file = gallery + d[0] + "/" + d[0] + "_00" + d[1] + ".xml";
            }
            if(d[2].size()==2)
            {
                xml_file1 = gallery + d[0] + "/" + d[0] + "_00" + d[2] + ".xml";
            }
            if(d[1].size()==3)
            {
                xml_file = gallery + d[0] + "/" + d[0] + "_0" + d[1] + ".xml";
            }
            if(d[2].size()==3)
            {
                xml_file1 = gallery + d[0] + "/" + d[0] + "_0" + d[2] + ".xml";
            }            
            cv::FileStorage fsread(xml_file,cv::FileStorage::READ);
            cv::Mat feature_vec;
            fsread["feature"]>>feature_vec;    
            cv::FileStorage fsread1(xml_file1,cv::FileStorage::READ);
            cv::Mat feature_vec1;
            fsread1["feature"]>>feature_vec1; 
            if(feature_vec.empty())
               {
                std::cout<<xml_file<<" is empty "<<std::endl;
                continue;
               }           
            if(feature_vec1.empty())
               {
               std::cout<<xml_file1<<" is empty "<<std::endl;
               continue;
               }    

            float dst =calculSimilar(feature_vec,feature_vec1);
            num_face +=1;
            if(dst >1.37)
               {
                std::cout<<"name:"<<s<<" || dst: "<<dst<<std::endl;  
                num+=1; 
                if(dst >1.6) 
                {
                    num_mtcnn +=1;
                }  
               }
      
        }
        else
        {
            std::string xml_file = gallery + d[0] + "/" + d[0] + "_000" + d[1] + ".xml";
            std::string xml_file1 = gallery + d[2] + "/"+d[2] +  "_000" + d[3] + ".xml";        
            if(d[1].size()==2)
            {
                xml_file = gallery + d[0] + "/" + d[0] + "_00" + d[1] + ".xml";
            }
            if(d[3].size()==2)
            {
                xml_file1 = gallery + d[2] + "/" + d[2] + "_00" + d[3] + ".xml";
            }
            if(d[1].size()==3)
            {
                xml_file = gallery + d[0] + "/" + d[0] + "_0" + d[1] + ".xml";
            }
            if(d[3].size()==3)
            {
                xml_file1 = gallery + d[2] + "/" + d[2] + "_0" + d[3] + ".xml";
            }            
            cv::FileStorage fsread(xml_file,cv::FileStorage::READ);
            cv::Mat feature_vec;
            fsread["feature"]>>feature_vec;    
            cv::FileStorage fsread1(xml_file1,cv::FileStorage::READ);
            cv::Mat feature_vec1;
            fsread1["feature"]>>feature_vec1; 
            if(feature_vec.empty())
               {
                std::cout<<xml_file<<" is empty "<<std::endl;
                continue;
               }           
            if(feature_vec1.empty())
               {
               std::cout<<xml_file1<<" is empty "<<std::endl;
               continue;
               }    

            float dst =calculSimilar(feature_vec,feature_vec1);
            num_face +=1;
            if(dst <1.37)
               {
                std::cout<<"name:"<<s<<" || dst: "<<dst<<std::endl;  
                num+=1;    
               }
        }
        
    }
    std::cout<<"error："<<num/num_face<<std::endl; 
    std::cout<<"accurate："<<1 - num/num_face<<std::endl;
    std::cout<<"有"<<num_mtcnn<<"张图像可能因定位错人脸而匹配失败"<<endl;
    infile.close();             //关闭文件输入流
}

void video_verification()
{
   
}

int main(int argc, char** argv) 
{
    const char *model_path = "../models";
    MTCNN mtcnn(model_path);
    Recognize recognize(model_path);

    if ((argc > 1) && !strcmp(argv[1], "-init_gallery")) {
        std::string gallery = argv[2];
        init_gallery(gallery,recognize,mtcnn);
        return 0;
    } else if ((argc > 1) && !strcmp(argv[1], "-gallery") && !strcmp(argv[3], "-pairs")) {
        std::string gallery = argv[2];
        std::string pairsfile = argv[3];
        lfw_test(pairsfile,gallery);
        return 0;
    }
    else if((argc > 1) && !strcmp(argv[1], "-gallery") && !strcmp(argv[3], "-verification"))
    {
        cv::string imgdir =argv[4];
        cv::Mat img = cv::imread(imgdir,1);
        if(img.empty())
        {
            cout<<"image read fail";
            return 0;
        }
        cv::string gallery = argv[2];
        vector<Personid> face_library;
        init_face_library(gallery,face_library);
        std::vector<match_result> results;
        faceimg_verification(img,face_library,mtcnn,recognize,results,1.37);
        if(results.empty())
        {
           cout<<"no face match."<<std::endl;
        }
        else
        {
            for (auto result:results)
            {
                cout<<"match success "<<result.name<<":"<<result.dst<<std::endl;
            }
            
        }
        

    }
    else if((argc > 1) && !strcmp(argv[1], "-image"))   
    {
        cv::string imgdir =argv[2];
        cv::Mat img = cv::imread(imgdir,1);
        if(img.empty())
        {
            cout<<"image read fail";
            return 0;
        } 
        cv::string gallery = "../gallery";
        vector<Personid> face_library;
        init_face_library(gallery,face_library);
        std::vector<match_result> results;
        faceimg_verification(img,face_library,mtcnn,recognize,results,1.37);
        if(results.empty())
        {
           cout<<"no face match."<<std::endl;
        }
        else
        {
            for (auto result:results)
            {
                cout<<"match success "<<result.name<<":"<<result.dst<<std::endl;
            }
            
        }              
    }
    video_verification();
    return 0;    
}

    //std::string gallery = "/home/coco/SecurityVision/ncnn-project/face_recognition/gallery";

    //std::string imgdir ="/home/coco/SecurityVision/Aaron_Eckhart_0001.jpg" ;
    //std::string xmldir = "/home/coco/SecurityVision/gallery.xml";
    //cv::Mat feature_vec;
    //cv::Mat img = cv::imread(imgdir,1);
    //recognize.start(img,feature_vec);    

    // std::string gallery = "/home/coco/SecurityVision/ncnn-project/face_recognition/gallery/face_feature/";
    // std::string pairsfile = "/home/coco/SecurityVision/pairs.txt";
    // lfw_test(pairsfile,gallery);


    //std::string face_path = "/home/coco/SecurityVision/111";
    //std::string aglin_path = "/home/coco/SecurityVision/222";
    //gallery_face_aglin(face_path,aglin_path);
    // std::string aglin_path = "/home/coco/SecurityVision/222";
    // std::string xml_path = "/home/coco/SecurityVision/333" ;
    // get_gallery_feature(aglin_path,xml_path);
