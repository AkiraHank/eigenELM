#include "elm_in_elm.h"
#include "functions.h"

#include <random>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>

ELM_IN_ELM::ELM_IN_ELM()
{
    
}

void ELM_IN_ELM::saveModel(std::string dirPath)
{
    //确保dirPath末尾是"/"
    if(dirPath[dirPath.length()-1] != '/'){
        dirPath.append("/");
    }
    
    //如果目标文件夹不存在则创建
    if(access(dirPath.data(),F_OK) == -1){
        int flag = mkdir(dirPath.data(),0777);
        if(flag == -1){
            std::cout<<"文件夹\""<<dirPath<<"\"不存在且创建失败！"<<std::endl;
            exit(1);
        }
    }
    
    //打开文件
    std::ofstream ofs(dirPath+"mainModel", std::ios::out|std::ios::binary);
    if(!ofs.is_open()){
        std::cout<<"打开或生成文件\""<<dirPath+"mainModel"<<"\"失败!"<<std::endl;
        exit(1);
    }
    
    //写入矩阵尺寸信息
    int params[3] = {m_nSubElms,m_nSubElmH,m_O};
    ofs.write((char*)params,sizeof(params));
    
    //写入权重矩阵
    int h = m_nSubElms;
    for(int i=0;i<h;i++)
        for(int j=0;j<m_O;j++){
            ofs.write((char*)&m_Who(i,j),sizeof(float));
        }
    for(int i=0;i<h;i++)
        for(int j=0;j<h;j++){
            ofs.write((char*)&m_K(i,j),sizeof(float));
        }
    
    ofs.close();
    
    //保存子模型
    for(int i=0;i<m_nSubElms;i++){
        m_subElms[i].saveModel(dirPath+"subModel"+std::to_string(i));
    }
}

void ELM_IN_ELM::loadModel(std::string dirPath)
{
    //确保dirPath末尾是"/"
    if(dirPath[dirPath.length()-1] != '/'){
        dirPath.append("/");
    }
    
    //打开文件
    std::ifstream ifs(dirPath+"mainModel",std::ios::in|std::ios::binary);
    if(!ifs.is_open()){
        std::cout<<"打开文件\""<<dirPath+"mainModel"<<"\"失败!"<<std::endl;
        exit(1);
    }
    
    int params[3];
    ifs.read((char*)params,sizeof(params));
    m_nSubElms = params[0];
    m_nSubElmH = params[1];
    m_O = params[2];
    
    int h = m_nSubElms;
    m_Who.resize(h,m_O);
    m_K.resize(h,h);
    for(int i=0;i<h;i++)
        for(int j=0;j<m_O;j++){
            ifs.read((char*)&m_Who(i,j),sizeof(float));
        }
    for(int i=0;i<h;i++)
        for(int j=0;j<h;j++){
            ifs.read((char*)&m_K(i,j),sizeof(float));
        }
    
    ifs.close();
    
    //加载子模型
    m_subElms.resize(m_nSubElms);
    for(int i=0;i<m_nSubElms;i++){
        m_subElms[i].loadModel(dirPath+"subModel"+std::to_string(i));
    }
}

void ELM_IN_ELM::setSubElmsNum(int n)
{
    m_nSubElms = n;
}

void ELM_IN_ELM::setSubModelHiddenNodes(int n)
{
    m_nSubElmH = n;
}

void ELM_IN_ELM::train(const Eigen::MatrixXf &featuresMat, const Eigen::MatrixXf &targetsMat)
{
    //初次训练的初始化
    if(m_subElms.empty()){
        m_O = targetsMat.cols();
        
        //初始化子elm
        m_subElms.resize(m_nSubElms);
        for(int i=0;i<m_nSubElms;i++){
            m_subElms[i].setHiddenNodes(m_nSubElmH);
            std::random_device rd;
            std::mt19937 mt(rd());
            m_subElms[i].setRandomState(mt());
        }
        
        //初始化Who
        m_Who.resize(m_nSubElms,m_O);
        m_Who.setZero();
        
        //初始化K
        m_K.resize(m_nSubElms,m_nSubElms);
        m_K.setZero();
    }
    
    //训练子elm
    for(int i=0;i<m_nSubElms;i++){
        m_subElms[i].train(featuresMat,targetsMat);
    }
    
    //得到子elm的输出
    std::vector<Eigen::MatrixXf> subElmOutputs(m_nSubElms);
    for(int i=0;i<m_nSubElms;i++){
        Eigen::MatrixXf tmpOut;
        m_subElms[i].predict(featuresMat,tmpOut);
        denseEncodeOutput(tmpOut,subElmOutputs[i]);
    }
    
    //拼接子elm的输出
    Eigen::MatrixXf H;
    H.resize(featuresMat.rows(),m_nSubElms);
    for(int i=0;i<m_nSubElms;i++){
        H.block(0,i,H.rows(),1) = subElmOutputs[i];
    }
    //sigmoid(H);
    
    //std::cout<<H<<std::endl;
    
    Eigen::MatrixXf U;
    elmsVote(H,m_O,U);
    std::cout<<"elm-in-elm 训练数据得分："<<calcScore(U,targetsMat)<<std::endl;
    
    /*
    //迭代更新K
    m_K = m_K + H.transpose()*H;
    
    //迭代更新Who
    m_Who = m_Who + pinv(m_K)*H.transpose()*(targetsMat-H*m_Who);
    
    //计算在训练数据上的准确率
    Eigen::MatrixXf U = H * m_Who;
    std::cout<<"elm-in-elm 训练数据得分："<<calcScore(U,targetsMat)<<std::endl;
    */
}

void ELM_IN_ELM::predict(const Eigen::MatrixXf &featuresMat, Eigen::MatrixXf &resultsMat)
{
    //得到子elm的输出
    std::vector<Eigen::MatrixXf> subElmOutputs(m_nSubElms);
    for(int i=0;i<m_nSubElms;i++){
        Eigen::MatrixXf tmpOut;
        m_subElms[i].predict(featuresMat,tmpOut);
        denseEncodeOutput(tmpOut,subElmOutputs[i]);
    }
    
    //拼接子elm的输出
    Eigen::MatrixXf H;
    H.resize(featuresMat.rows(),m_nSubElms);
    for(int i=0;i<m_nSubElms;i++){
        H.block(0,i,H.rows(),1) = subElmOutputs[i];
    }
    //sigmoid(H);
    
    elmsVote(H,m_O,resultsMat);
}

float ELM_IN_ELM::validate(const Eigen::MatrixXf &featuresMat, const Eigen::MatrixXf &targetsMat)
{
    //
    for(int i=0;i<m_subElms.size();i++){
        float score = m_subElms[i].validate(featuresMat,targetsMat);
        std::cout<<"subElm score:"<<score<<std::endl;
    }
    
    //
    Eigen::MatrixXf output;
    predict(featuresMat,output);
    float score = calcScore(output,targetsMat);
    return score;
}
