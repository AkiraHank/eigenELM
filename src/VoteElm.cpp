#include "VoteElm.h"
#include "functions.h"

#include <random>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include "fileUtils.hpp"

VoteElm::VoteElm()
{
    
}

//保存
void VoteElm::saveModel(std::string dirPath)
{
    //确保dirPath末尾是"/"
    if(dirPath[dirPath.length()-1] != '/'){
        dirPath.append("/");
    }
    
    //如果目标文件夹不存在则创建
    if(!fileUtils::checkInputFileValid(dirPath)) {
        int flag = fs::create_directory(dirPath);
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
    
    //写入elm个数和输出向量维数
    int params[2] = {m_nElms,m_O};
    ofs.write((char*)params,sizeof(params));
    
    ofs.close();
    
    //保存子模型
    for(int i=0;i<m_nElms;i++){
        m_elms[i].saveModel(dirPath+"subModel"+std::to_string(i));
    }
}

//加载
void VoteElm::loadModel(std::string dirPath)
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
    
    //加载elm个数和输出向量维数
    int params[2];
    ifs.read((char*)params,sizeof(params));
    m_nElms = params[0];
    m_O = params[1];
    
    ifs.close();
    
    //加载子模型
    m_elms.resize(m_nElms);
    for(int i=0;i<m_nElms;i++){
        m_elms[i].loadModel(dirPath+"subModel"+std::to_string(i));
    }
}

//设置elm个数
void VoteElm::setElmNum(int n)
{
    m_nElms = n;
}

//设置elm隐藏层节点数
void VoteElm::setElmHiddenNodes(int n)
{
    m_nElmH = n;
}

//训练
void VoteElm::train(const Eigen::MatrixXf &featuresMat, const Eigen::MatrixXf &targetsMat)
{
    //初次训练的初始化
    if(m_elms.empty()){
        m_O = targetsMat.cols();
        
        //初始化elm
        m_elms.resize(m_nElms);
        for(int i=0;i<m_nElms;i++){
            m_elms[i].setHiddenNodes(m_nElmH);
            std::random_device rd;
            std::mt19937 mt(rd());
            m_elms[i].setRandomState(mt());
        }
    }
    
    //训练elm
    for(int i=0;i<m_nElms;i++){
        m_elms[i].train(featuresMat,targetsMat);
    }
    
    //得到elm的输出
    std::vector<Eigen::MatrixXf> subElmOutputs(m_nElms);
    for(int i=0;i<m_nElms;i++){
        Eigen::MatrixXf tmpOut;
        m_elms[i].predict(featuresMat,tmpOut);
        denseEncodeOutput(tmpOut,subElmOutputs[i]);
    }
    
    //拼接elm的输出
    Eigen::MatrixXf H;
    H.resize(featuresMat.rows(),m_nElms);
    for(int i=0;i<m_nElms;i++){
        H.block(0,i,H.rows(),1) = subElmOutputs[i];
    }
    
    Eigen::MatrixXf U;
    elmsVote(H,m_O,U);
    std::cout<<"elms vote 训练数据得分："<<calcScore(U,targetsMat)<<std::endl;
}

//预测
void VoteElm::predict(const Eigen::MatrixXf &featuresMat, Eigen::MatrixXf &resultsMat)
{
    //得到子elm的输出
    std::vector<Eigen::MatrixXf> subElmOutputs(m_nElms);
    for(int i=0;i<m_nElms;i++){
        Eigen::MatrixXf tmpOut;
        m_elms[i].predict(featuresMat,tmpOut);
        denseEncodeOutput(tmpOut,subElmOutputs[i]);
    }
    
    //拼接子elm的输出
    Eigen::MatrixXf H;
    H.resize(featuresMat.rows(),m_nElms);
    for(int i=0;i<m_nElms;i++){
        H.block(0,i,H.rows(),1) = subElmOutputs[i];
    }
    
    elmsVote(H,m_O,resultsMat);
}

//计分
float VoteElm::validate(const Eigen::MatrixXf &featuresMat, const Eigen::MatrixXf &targetsMat)
{
    //elm得分
    for(int i=0;i<m_elms.size();i++){
        float score = m_elms[i].validate(featuresMat,targetsMat);
        std::cout<<"elm["<<std::to_string(i)<<"] score:"<<score<<std::endl;
    }
    
    //vote-elm得分
    Eigen::MatrixXf output;
    predict(featuresMat,output);
    float score = calcScore(output,targetsMat);
    return score;
}
