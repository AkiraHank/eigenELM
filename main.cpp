#include <iostream>
#include <random>

#include <Eigen>
#include "functions.h"
#include "elm_in_elm.h"

int main(int argc, char ** argv)
{
    if(argc > 1 && (strcmp(argv[1],"validate")==0)){
        std::string modelDir = argv[2];
        std::string inputFile = argv[3];
        
        //加载模型
        ELM_IN_ELM eie;
        eie.loadModel(modelDir);
        
        //加载数据
        Eigen::MatrixXf featuresMat;
        Eigen::MatrixXf targetsMat;
        std::vector<std::string> id_label_list;
        if(modelDir[modelDir.length()-1] != '/')
            modelDir.append("/");
        loadLabelList(modelDir+"id_label_list.txt",id_label_list);
        readValData(inputFile,id_label_list,featuresMat,targetsMat);
        //normFeatures(featuresMat,-1,1);
        
        //测试得分
        float score = eie.validate(featuresMat,targetsMat);
        std::cout<<"elm-in-elm测试数据得分："<<score<<std::endl;
        
        return 0;
    }
    
    if(argc == 5){ //从零开始训练
        int nSubElms = atoi(argv[1]);
        int nSubElmH = atoi(argv[2]);
        std::string inputFile = argv[3];
        std::string modelDir = argv[4];
        
        //读入训练数据
        Eigen::MatrixXf featuresMat;
        Eigen::MatrixXf targetsMat;
        std::vector<std::string> id_label_list;
        readTrainData(inputFile,featuresMat,targetsMat,id_label_list);
        
        //训练并保存模型
        ELM_IN_ELM eie;
        eie.setSubElmsNum(nSubElms);
        eie.setSubModelHiddenNodes(nSubElmH);
        eie.train(featuresMat,targetsMat);
        eie.saveModel(modelDir);
        
        //保存预测用的id-label键值对
        if(modelDir[modelDir.length()-1] != '/')
            modelDir.append("/");
        saveLabelList(modelDir+"id_label_list.txt",id_label_list);
    }
    else if(argc == 3){ //加载模型继续训练
        std::string inputFile = argv[1];
        std::string modelDir = argv[2];
        
        //读入训练数据
        Eigen::MatrixXf featuresMat;
        Eigen::MatrixXf targetsMat;
        std::vector<std::string> id_label_list;
        readTrainData(inputFile,featuresMat,targetsMat,id_label_list);
        
        //加载、训练并保存模型
        ELM_IN_ELM eie;
        eie.loadModel(modelDir);
        eie.train(featuresMat,targetsMat);
        eie.saveModel(modelDir);
        
        //保存预测用的id-label键值对
        if(modelDir[modelDir.length()-1] != '/')
            modelDir.append("/");
        saveLabelList(modelDir+"id_label_list.txt",id_label_list);
    }
    else if(argc == 4){ //预测
        std::string inputFile = argv[1];
        std::string modelDir = argv[2];
        std::string outputFile = argv[3];
        
        //读入特征
        Eigen::MatrixXf featuresMat;
        readFeature(inputFile,featuresMat);
        
        //加载模型并预测
        ELM_IN_ELM eie;
        eie.loadModel(modelDir);
        Eigen::MatrixXf output;
        eie.predict(featuresMat,output);
        
        //加载id-label键值对
        std::vector<std::string> id_label_list;
        loadLabelList("./net/id_label_list.txt",id_label_list);
        
        //转换成label并输出到文件
        std::vector<std::string> outputLabels;
        for(int i=0;i<output.rows();i++){
            int id = getRowMaxId(output.row(i));
            outputLabels.push_back(id_label_list[id]);
        }
        saveLabelList(outputFile,outputLabels);
    }
    else{
        std::cout<<"参数错误!"<<std::endl;
        exit(1);
    }
    
    return 0;
}
