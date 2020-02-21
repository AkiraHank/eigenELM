#ifndef ELM_H
#define ELM_H

#include <Eigen>

class ELM
{
public:
    ELM();
    
    //保存
    void saveModel(std::string filePath);
    
    //加载
    void loadModel(std::string filePath);
    
    //设置隐藏层节点数
    void setHiddenNodes(int hiddenNodes);
    
    //设置随机种子
    void setRandomState(int randomState);
    
    //训练
    void train(const Eigen::MatrixXf &featuresMat, const Eigen::MatrixXf &targetsMat);
    
    //预测
    void predict(const Eigen::MatrixXf &featuresMat, Eigen::MatrixXf &resultsMat);
    
    //计分
    float validate(const Eigen::MatrixXf &featuresMat, const Eigen::MatrixXf &targetsMat);
    
private:
    //随机种子
    int m_randomState;
    
    //输入层节点数
    int m_I;
    
    //隐藏层节点数
    int m_H;
    
    //输出层节点数
    int m_O;
    
    //权重
    Eigen::MatrixXf m_Wih;
    Eigen::MatrixXf m_Who;
    
    //在线序列学习中用到的，保留了历史数据的一个矩阵。等于H.t()*H
    Eigen::MatrixXf m_K;
};

#endif // ELM_H
