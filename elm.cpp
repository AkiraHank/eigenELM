#include "elm.h"
#include "functions.h"

#include <random>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>

ELM::ELM()
{
    
}

void ELM::saveModel(std::string filePath)
{
    //打开文件
    std::ofstream ofs(filePath, std::ios::out|std::ios::binary);
    if(!ofs.is_open()){
        std::cout<<"打开或生成文件\""<<filePath<<"\"失败!"<<std::endl;
        exit(1);
    }
    
    //写入矩阵尺寸信息
    int params[3] = {m_I,m_H,m_O};
    ofs.write((char*)params,sizeof(params));
    
    //写入Wih,Who
    for(int i=0;i<m_I;i++)
        for(int j=0;j<m_H;j++){
            ofs.write((char*)&m_Wih(i,j),sizeof(float));
        }
    for(int i=0;i<m_H;i++)
        for(int j=0;j<m_H;j++){
            ofs.write((char*)&m_K(i,j),sizeof(float));
        }
    for(int i=0;i<m_H;i++)
        for(int j=0;j<m_O;j++){
            ofs.write((char*)&m_Who(i,j),sizeof(float));
        }
    
    //关闭文件
    ofs.close();
}

void ELM::loadModel(std::string filePath)
{
    //打开文件
    std::ifstream ifs(filePath,std::ios::in|std::ios::binary);
    if(!ifs.is_open()){
        std::cout<<"打开文件\""<<filePath<<"\"失败!"<<std::endl;
        exit(1);
    }
    
    int params[3];
    ifs.read((char*)params,sizeof(params));
    m_I = params[0];
    m_H = params[1];
    m_O = params[2];
    
    m_Wih.resize(m_I,m_H);
    m_K.resize(m_H,m_H);
    m_Who.resize(m_H,m_O);
    for(int i=0;i<m_I;i++)
        for(int j=0;j<m_H;j++){
            ifs.read((char*)&m_Wih(i,j),sizeof(float));
        }
    for(int i=0;i<m_H;i++)
        for(int j=0;j<m_H;j++){
            ifs.read((char*)&m_K(i,j),sizeof(float));
        }
    for(int i=0;i<m_H;i++)
        for(int j=0;j<m_O;j++){
            ifs.read((char*)&m_Who(i,j),sizeof(float));
        }
    
    ifs.close();
}

void ELM::setHiddenNodes(int hiddenNodes)
{
    m_H = hiddenNodes;
}

void ELM::setRandomState(int randomState)
{
    m_randomState = randomState;
}

void ELM::train(const Eigen::MatrixXf &featuresMat, const Eigen::MatrixXf &targetsMat)
{
    m_I = featuresMat.cols();
    m_O = targetsMat.cols();
    
    //初次训练的初始化
    if(m_Wih.rows() == 0){
        //随机生成Wih
        genRandomMat(m_Wih,m_I,m_H,-1.0,1.0,m_randomState);
        
        //初始化Who
        m_Who.resize(m_H,m_O);
        m_Who.setZero();
        
        //初始化K
        m_K.resize(m_H,m_H);
        m_K.setZero();
    }
    
    //计算隐藏层输出
    Eigen::MatrixXf H = featuresMat * m_Wih;
    sigmoid(H);
    H.block(0,H.cols()-1,H.rows(),1).setOnes();
    
    //迭代更新K
    m_K = m_K + H.transpose()*H;
    
    //迭代更新Who
    m_Who = m_Who + pinv(m_K)*H.transpose()*(targetsMat-H*m_Who);
    
    //计算在训练数据上的准确率
    //Eigen::MatrixXf U = H * m_Who;
    //std::cout<<"训练数据得分："<<calcScore(U,targetsMat)<<std::endl;
}

void ELM::predict(const Eigen::MatrixXf &featuresMat, Eigen::MatrixXf &resultsMat)
{
    resultsMat = featuresMat * m_Wih;
    sigmoid(resultsMat);
    resultsMat.block(0,resultsMat.cols()-1,resultsMat.rows(),1).setOnes();
    
    resultsMat = resultsMat * m_Who;
}

void ELM::validate(const Eigen::MatrixXf &featuresMat, const Eigen::MatrixXf &targetsMat)
{
    Eigen::MatrixXf m;
    predict(featuresMat,m);
    
    float score = calcScore(m,targetsMat);
    std::cout<<"测试数据得分："<<score<<std::endl;
}
