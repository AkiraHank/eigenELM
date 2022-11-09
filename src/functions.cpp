#include "functions.h"

#include <iostream>
#include <fstream>
#include <map>
#include <random>

void readTrainData(std::string filePath, Eigen::MatrixXf &featuresMat, 
                   Eigen::MatrixXf &targetsMat, //与features对应的target
                   std::vector<std::string> &id_label_list) //label列表，能够直接根据id得到label
{
    std::ifstream ifs(filePath,std::ios::in);
    if(!ifs.is_open()){
        std::cout<<"文件 "<<filePath<<" 打开失败!"<<std::endl;
        exit(1);
    }
    
    std::map<std::string,int> targetMap;
    std::vector<std::string> targetStrList;
    std::vector<std::vector<float>> featureVecs;
    
    std::string line;
    while(getline(ifs,line)){
        //去除换行符
        if(line[line.length()-1] == '\n'){
            line = line.substr(0,line.length()-1);
        }
        
        //找到分隔符
        int colonPos = line.find_first_of(":");
        if(colonPos == std::string::npos){
            std::cout<<"文件\\"<<filePath<<"\\的内容格式不正确!"<<std::endl;
            exit(1);
        }
        
        //分隔符前为target
        targetStrList.push_back(line.substr(0,colonPos));
        targetMap.insert(std::make_pair(line.substr(0,colonPos),0));
        
        //分隔符后为feature
        std::string featureStr = line.substr(colonPos+1,line.length());
        std::vector<float> featureVec;
        while(1){
            int commaPos = featureStr.find_first_of(",");
            if(commaPos == std::string::npos){
                break;
            }
            float f = atof(featureStr.substr(0,commaPos).data());
            featureVec.push_back(f);
            
            featureStr = featureStr.substr(commaPos+1,featureStr.length());
        }
        featureVec.push_back(atof(featureStr.data()));
        featureVecs.push_back(featureVec);
    }
    
    ifs.close();
    
    //feature转为MatrixXf格式
    featuresMat.resize(featureVecs.size(),featureVecs[0].size());
    for(int i=0;i<featuresMat.rows();i++)
        for(int j=0;j<featuresMat.cols();j++){
            featuresMat(i,j) = featureVecs[i][j];
        }
    
    //target转为MatrixXf格式
    targetsMat.resize(targetStrList.size(),targetMap.size());
    int id=0;
    for(std::map<std::string,int>::iterator it=targetMap.begin();it!=targetMap.end();it++){
        it->second = id;
        id++;
        
        id_label_list.push_back(it->first);
    }
    for(int i=0;i<targetStrList.size();i++){
        for(int j=0;j<targetsMat.cols();j++){
            targetsMat(i,j) = 0;
        }
        int code = targetMap[targetStrList[i]];
        targetsMat(i,code) = 1;
    }
}

void readValData(std::string filePath, 
                 std::vector<std::string> id_label_list, 
                 Eigen::MatrixXf &featuresMat, 
                 Eigen::MatrixXf &targetsMat)
{
    std::ifstream ifs(filePath,std::ios::in);
    if(!ifs.is_open()){
        std::cout<<"文件\\"<<filePath<<"\\打开失败!"<<std::endl;
        exit(1);
    }
    
    //由输入的id_label_list生成targetMap
    std::map<std::string,int> targetMap;
    for(int i=0;i<id_label_list.size();i++){
        targetMap.insert(std::make_pair(id_label_list[i],i));
    }
    
    //读入feature和targetStrList
    std::vector<std::string> targetStrList;
    std::vector<std::vector<float>> featureVecs;
    
    std::string line;
    while(getline(ifs,line)){
        //去除换行符
        if(line[line.length()-1] == '\n'){
            line = line.substr(0,line.length()-1);
        }
        
        //找到分隔符
        int colonPos = line.find_first_of(":");
        if(colonPos == std::string::npos){
            std::cout<<"文件\\"<<filePath<<"\\的内容格式不正确!"<<std::endl;
            exit(1);
        }
        
        //分隔符前为target
        targetStrList.push_back(line.substr(0,colonPos));
        
        //分隔符后为feature
        std::string featureStr = line.substr(colonPos+1,line.length());
        std::vector<float> featureVec;
        while(1){
            int commaPos = featureStr.find_first_of(",");
            if(commaPos == std::string::npos){
                break;
            }
            float f = atof(featureStr.substr(0,commaPos).data());
            featureVec.push_back(f);
            
            featureStr = featureStr.substr(commaPos+1,featureStr.length());
        }
        featureVec.push_back(atof(featureStr.data()));
        featureVecs.push_back(featureVec);
    }
    
    ifs.close();
    
    //feature转为MatrixXf格式
    featuresMat.resize(featureVecs.size(),featureVecs[0].size());
    for(int i=0;i<featuresMat.rows();i++)
        for(int j=0;j<featuresMat.cols();j++){
            featuresMat(i,j) = featureVecs[i][j];
        }
    
    //target转为MatrixXf格式
    targetsMat.resize(targetStrList.size(),targetMap.size());
    for(int i=0;i<targetStrList.size();i++){
        for(int j=0;j<targetsMat.cols();j++){
            targetsMat(i,j) = 0;
        }
        int code = targetMap[targetStrList[i]];
        targetsMat(i,code) = 1;
    }
}

void readFeature(std::string filePath, Eigen::MatrixXf &featuresMat)
{
    std::ifstream ifs(filePath,std::ios::in);
    if(!ifs.is_open()){
        std::cout<<"文件\\"<<filePath<<"\\打开失败!"<<std::endl;
        exit(1);
    }
    
    std::vector<std::vector<float>> featureVecs;
    std::string line;
    while(getline(ifs,line)){
        //去除换行符
        if(line[line.length()-1] == '\n'){
            line = line.substr(0,line.length()-1);
        }
        
        std::vector<float> featureVec;
        while(1){
            int commaPos = line.find_first_of(",");
            if(commaPos == std::string::npos){
                break;
            }
            float f = atof(line.substr(0,commaPos).data());
            featureVec.push_back(f);
            
            line = line.substr(commaPos+1,line.length());
        }
        featureVec.push_back(atof(line.data()));
        featureVecs.push_back(featureVec);
    }
    
    ifs.close();
    
    //feature转为MatrixXf格式
    featuresMat.resize(featureVecs.size(),featureVecs[0].size());
    for(int i=0;i<featuresMat.rows();i++)
        for(int j=0;j<featuresMat.cols();j++){
            featuresMat(i,j) = featureVecs[i][j];
        }
}

Eigen::MatrixXf pinv(Eigen::MatrixXf A)
{
    //M=USV*
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    double  pinvtoler = 1.e-8; //tolerance
    int row = A.rows();
    int col = A.cols();
    int k = std::min(row,col);
    Eigen::MatrixXf X = Eigen::MatrixXf::Zero(col,row);
    Eigen::MatrixXf singularValues_inv = svd.singularValues();//奇异值
    Eigen::MatrixXf singularValues_inv_mat = Eigen::MatrixXf::Zero(col, row);
    for (long i = 0; i<k; ++i) {
        if (singularValues_inv(i) > pinvtoler)
            singularValues_inv(i) = 1.0 / singularValues_inv(i);
        else singularValues_inv(i) = 0;
    }
    for (long i = 0; i < k; ++i) {
        singularValues_inv_mat(i, i) = singularValues_inv(i);
    }
    X=(svd.matrixV())*(singularValues_inv_mat)*(svd.matrixU().transpose());//X=VS+U*

    return X;
}

void genRandomMat(Eigen::MatrixXf &mat, int rows, int cols, 
                  float lowerLimit, float upperLimit, int randomState)
{
    std::srand(randomState);
    
    mat.resize(rows,cols);
    for(int i=0;i<mat.rows();i++)
        for(int j=0;j<mat.cols();j++){
            mat(i,j) = lowerLimit + (std::rand()%10000)/float(10000)*(upperLimit-lowerLimit);
        }
}

void sigmoid(Eigen::MatrixXf &mat)
{
    for(int i=0;i<mat.rows();i++)
        for(int j=0;j<mat.cols();j++){
            mat(i,j) = 1/(1+std::exp(-mat(i,j)));
        }
}

int getRowMaxId(Eigen::MatrixXf row)
{
    int maxId=0;
    float maxVal = row(0,0);
    for(int j=0;j<row.cols();j++){
        if(row(0,j) > maxVal){
            maxVal = row(0,j);
            maxId = j;
        }
    }
    
    return maxId;
}

float calcScore(const Eigen::MatrixXf &output, const Eigen::MatrixXf &target)
{
    int Q = output.rows();
    int ncorrect = 0;
    
    for(int i=0;i<Q;i++){
        int targetMaxId = getRowMaxId(target.row(i));
        int outMaxId = getRowMaxId(output.row(i));
        
        if(targetMaxId == outMaxId)
            ncorrect++;
    }
    
    return ncorrect/float(Q);
}

void saveLabelList(std::string filePath, std::vector<std::string> id_label_list)
{
    std::string contentStr = "";
    for(int i=0;i<id_label_list.size();i++){
        contentStr += id_label_list[i] + "\n";
    }
    contentStr = contentStr.substr(0,contentStr.length()-1);
    
    std::ofstream ofs(filePath,std::ios::out);
    if(!ofs.is_open()){
        std::cout<<"文件\\"<<filePath<<"\\打开失败!"<<std::endl;
        exit(1);
    }
    ofs << contentStr;
    
    ofs.close();
}

void loadLabelList(std::string filePath, std::vector<std::string> &id_label_list)
{
    std::ifstream ifs(filePath,std::ios::in);
    if(!ifs.is_open()){
        std::cout<<"文件\\"<<filePath<<"\\打开失败!"<<std::endl;
        exit(1);
    }
    
    std::string line;
    while(getline(ifs,line)){
        //去除换行符
        if(line[line.length()-1] == '\n'){
            line = line.substr(0,line.length()-1);
        }
        
        id_label_list.push_back(line);
    }
    
    ifs.close();
}

void getMatMinMaxVal(Eigen::MatrixXf mat, float &minVal, float &maxVal)
{
    int rows = mat.rows();
    int cols = mat.cols();
    
    minVal = mat(0,0);
    maxVal = mat(0,0);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            float val = mat(i,j);
            if(val > maxVal)
                maxVal = val;
            else if(val < minVal)
                minVal = val;
        }
    }
}

void normFeatures(Eigen::MatrixXf &featuresMat, float lowerLimit, float upperLimit)
{
    float trange = upperLimit - lowerLimit;
    
    float minVal, maxVal;
    getMatMinMaxVal(featuresMat,minVal,maxVal);
    float srange = maxVal - minVal;
    float scaleRatio = trange/float(srange);
    
    int rows = featuresMat.rows();
    int cols = featuresMat.cols();
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            float val = featuresMat(i,j);
            featuresMat(i,j) = (val-minVal)*scaleRatio + lowerLimit;
        }
    }
}

void denseEncodeOutput(const Eigen::MatrixXf &mat, Eigen::MatrixXf &result)
{
    result.resize(mat.rows(),1);
    
    for(int r=0;r<mat.rows();r++){
        int maxId = getRowMaxId(mat.row(r));
        result(r,0) = maxId;
    }
}

void elmsVote(const Eigen::MatrixXf &input, int outDim, Eigen::MatrixXf &output)
{
    int nsamples = input.rows();
    int nelms = input.cols();
    
    output.resize(nsamples,outDim);
    output.setZero();
    
    for(int i=0;i<nsamples;i++){
        std::vector<int> scoreBox(outDim,0);
        for(int j=0;j<nelms;j++){
            scoreBox[int(input(i,j))]++;
        }
        
        int maxScore=0;
        int maxId=0;
        for(int m=0;m<outDim;m++){
            if(scoreBox[m] > maxScore){
                maxScore = scoreBox[m];
                maxId = m;
            }
        }
        
        output(i,maxId) = 1;
    }
}
