#ifndef VOTEELM_H
#define VOTEELM_H

#include "elm.h"
#include <Eigen>

class VoteElm
{
public:
    VoteElm();
    
    //保存
    void saveModel(std::string dirPath);
    
    //加载
    void loadModel(std::string dirPath);
    
    //设置子elm个数
    void setElmNum(int n);
    
    //设置子elm隐藏层节点数
    void setElmHiddenNodes(int n);
    
    //训练
    void train(const Eigen::MatrixXf &featuresMat, const Eigen::MatrixXf &targetsMat);
    
    //预测
    void predict(const Eigen::MatrixXf &featuresMat, Eigen::MatrixXf &resultsMat);
    
    //计分
    float validate(const Eigen::MatrixXf &featuresMat, const Eigen::MatrixXf &targetsMat);
    
private:
    std::vector<ELM> m_elms;
    
    int m_nElms;
    
    int m_nElmH;
    
    //输出向量维数
    int m_O;
};

#endif // VOTEELM_H
