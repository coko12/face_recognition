该项目运行系统为ubuntu，win系统运行需要局部修改

ubuntu运行步骤：
ncnn编译
1.cd ncnn/build
2.cmake ..
3.make -j8

face项目编译
1.cd face_recognition/build
2.cmake ..
3.make

onnx转化：
1.下载onnx模型
2.onnx.py文件
3.复制onnx模型到ncnn/build/tools/onnx文件夹中 运行 ./onnx2ncnn IR_50.onnx 生成ncnn模型文件。然后重命名
4.复制重命名的ncnn文件至face_recognition/models文件夹 如文件目录所示

算法运行：
1.复制lwf数据集 到face_recogntion/gallery/face_image 注意复制后文件格式为face_image/名字/人脸.jpg
2.cd face_recognition/build 
3.  ./facenet -init_gallery ../gallery 生成人脸对齐文件及人脸特征文件
4.  ./facenet -gallery ../gallery -pairs pairs.txt 测试lfw数据集


人脸匹配：
1.按上述1步骤到3步骤制作个人人脸识别库
2. ./facenet -image xxx.jpg 进行人脸匹配

