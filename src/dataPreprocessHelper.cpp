#include "DataPreprocessHelper.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

bool DataPreprocessHelper::initTrainInput(std::string const &_rawDataPath, std::string const &_targetPath) {
  // read rawData
  std::string rawDataPath = fs::current_path().string() + _rawDataPath;
  std::string targetPath = fs::current_path().string() + _targetPath;
  if (fileUtils::checkInputFileValid(rawDataPath) && fileUtils::checkOutputFileValid(targetPath)) {
    std::ifstream ifs(rawDataPath, std::ios::in);
    std::ofstream ifs1(targetPath, std::ios::out);
    if (!ifs.is_open()) {
      std::cout << "打开文件 " << rawDataPath << " 失败!" << std::endl;
      return false;
    }
    if (!ifs1.is_open()) {
      std::cout << "打开文件 " << targetPath << " 失败!" << std::endl;
      return false;
    }

    std::string strLine;
    std::unordered_map<std::string, int> cate2num;
    while (getline(ifs, strLine)) {
      for (auto &c : strLine) {
        if (c == ',') {
          c = ' ';
        }
      }

      std::vector<float> nums(4);
      std::stringstream ss;
      ss << strLine;
      for (int i = 0; i < 4; i++) {
        ss >> nums[i];
      }
      printf("\n");

      ss >> strLine;
      if (!cate2num.count(strLine)) {
        cate2num.insert({strLine, cate2num.size()});
      }
      strLine = std::to_string(cate2num[strLine]) + ":";
      for (auto t : nums) {
        strLine += std::to_string(t) + ",";
      }
      strLine.pop_back();

      ifs1 << strLine << std::endl;
    }
    ifs.close();

  } else {
    std::cout << "输入文件 " + rawDataPath + " 不存在\n";
    return false;
  }
}
