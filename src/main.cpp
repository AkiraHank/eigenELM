#include <iostream>
#include <random>

#include "functions.h"
#include "VoteElm.h"
#include "DataPreprocessHelper.h"
using std::cout;

int main(int argc, char** argv) {
  if (argc > 1 && (strcmp(argv[1], "validate") == 0)) {
    std::string modelDir = argv[2];
    std::string inputFile = argv[3];

    //加载模型
    VoteElm velm;
    velm.loadModel(modelDir);

    //加载数据
    Eigen::MatrixXf featuresMat;
    Eigen::MatrixXf targetsMat;
    std::vector<std::string> id_label_list;
    if (modelDir[modelDir.length() - 1] != '/')
      modelDir.append("/");
    loadLabelList(modelDir + "id_label_list.txt", id_label_list);
    readValData(inputFile, id_label_list, featuresMat, targetsMat);

    //测试得分
    float score = velm.validate(featuresMat, targetsMat);
    std::cout << "vote-elm 测试数据得分：" << score << std::endl;

    return 0;
  }

  if (argc == 5) { //从零开始训练

    int nSubElms = atoi(argv[1]);
    int nSubElmH = atoi(argv[2]);
    std::string inputFile = argv[3];
    std::string modelDir = argv[4];
    inputFile = fs::current_path().string() + inputFile;
    modelDir = fs::current_path().string() + modelDir;

    auto& dataPreprocessHelper = DataPreprocessHelper::getInstance();
    dataPreprocessHelper.initTrainInput("/train/iris.data", "/train/train_input.txt");

    //读入训练数据
    Eigen::MatrixXf featuresMat;
    Eigen::MatrixXf targetsMat;
    std::vector<std::string> id_label_list;
    readTrainData(inputFile, featuresMat, targetsMat, id_label_list);

    //训练并保存模型
    VoteElm velm;
    velm.setElmNum(nSubElms);
    velm.setElmHiddenNodes(nSubElmH);
    velm.train(featuresMat, targetsMat);
    velm.saveModel(modelDir);

    //保存预测用的id-label键值对
    if (modelDir[modelDir.length() - 1] != '/')
      modelDir.append("/");
    saveLabelList(modelDir + "id_label_list.txt", id_label_list);
  } else if (argc == 3) { //加载模型继续训练
    std::string inputFile = argv[1];
    std::string modelDir = argv[2];

    //读入训练数据
    Eigen::MatrixXf featuresMat;
    Eigen::MatrixXf targetsMat;
    std::vector<std::string> id_label_list;
    readTrainData(inputFile, featuresMat, targetsMat, id_label_list);

    //加载、训练并保存模型
    VoteElm velm;
    velm.loadModel(modelDir);
    velm.train(featuresMat, targetsMat);
    velm.saveModel(modelDir);

    //保存预测用的id-label键值对
    if (modelDir[modelDir.length() - 1] != '/')
      modelDir.append("/");
    saveLabelList(modelDir + "id_label_list.txt", id_label_list);
  } else if (argc == 4) { //预测
    std::string inputFile = argv[1];
    std::string modelDir = argv[2];
    std::string outputFile = argv[3];

    //读入特征
    Eigen::MatrixXf featuresMat;
    readFeature(inputFile, featuresMat);

    //加载模型并预测
    VoteElm velm;
    velm.loadModel(modelDir);
    Eigen::MatrixXf output;
    velm.predict(featuresMat, output);

    //加载id-label键值对
    if (modelDir[modelDir.length() - 1] != '/')
      modelDir.append("/");
    std::vector<std::string> id_label_list;
    loadLabelList(modelDir + "id_label_list.txt", id_label_list);

    //转换成label并输出到文件
    std::vector<std::string> outputLabels;
    for (int i = 0; i < output.rows(); i++) {
      int id = getRowMaxId(output.row(i));
      outputLabels.push_back(id_label_list[id]);
    }
    saveLabelList(outputFile, outputLabels);
  } else {
    std::cout << "参数错误!" << std::endl;
    exit(1);
  }

  return 0;
}
