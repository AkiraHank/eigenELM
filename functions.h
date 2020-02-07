#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <Eigen>

//从文件读入训练数据
void readTrainData(std::string filePath, Eigen::MatrixXf &featuresMat, 
                   Eigen::MatrixXf &targetsMat, 
                   std::vector<std::string> &id_label_list);

//从文件读入特征
void readFeature(std::string filePath, Eigen::MatrixXf &featuresMat);

//求广义逆
Eigen::MatrixXf pinv(Eigen::MatrixXf A);

//生成随机矩阵
void genRandomMat(Eigen::MatrixXf &mat, int rows, int cols, 
                  float lowerLimit, float upperLimit, int randomState);

//sigmoid激活函数
void sigmoid(Eigen::MatrixXf &mat);

//找到一行中的最大值id
int getRowMaxId(Eigen::MatrixXf row);

//根据输出向量和target向量计分
float calcScore(const Eigen::MatrixXf &output, const Eigen::MatrixXf &target);

//保存、读取string键值与输出向量的对应关系
void saveLabelList(std::string filePath, std::vector<std::string> id_label_list);
void loadLabelList(std::string filePath, std::vector<std::string> &id_label_list);

#endif // FUNCTIONS_H
