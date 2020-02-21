#ifndef ELM_IN_ELM_H
#define ELM_IN_ELM_H

#include <Eigen>
#include "elm.h"

class ELM_IN_ELM
{
public:
    ELM_IN_ELM();
    
    //保存
    void saveModel(std::string dirPath);
    
    //加载
    void loadModel(std::string dirPath);
    
    //设置子elm个数
    void setSubElmsNum(int n);
    
    //设置子elm隐藏层节点数
    void setSubModelHiddenNodes(int n);
    
    //训练
    void train(const Eigen::MatrixXf &featuresMat, const Eigen::MatrixXf &targetsMat);
    
    //预测
    void predict(const Eigen::MatrixXf &featuresMat, Eigen::MatrixXf &resultsMat);
    
    //计分
    float validate(const Eigen::MatrixXf &featuresMat, const Eigen::MatrixXf &targetsMat);
    
private:
    //子elm
    std::vector<ELM> m_subElms;
    
    //输出侧权重
    Eigen::MatrixXf m_Who;
    
    //子elm数目
    int m_nSubElms;
    
    //子elm隐藏层节点数
    int m_nSubElmH;
    
    //输出向量维数
    int m_O;
    
    //在线序列学习用到的
    Eigen::MatrixXf m_K;
};

#endif // ELM_IN_ELM_H
