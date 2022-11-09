#ifndef DATAPREPROCESSHELPER_H
#define DATAPREPROCESSHELPER_H

#include <string>
#include "fileUtils.hpp"

class DataPreprocessHelper {
public:
  DataPreprocessHelper(const DataPreprocessHelper &) = delete;
  DataPreprocessHelper &operator=(const DataPreprocessHelper &) = delete;

  static DataPreprocessHelper& getInstance() {
    static DataPreprocessHelper instance;
    return instance; 
  }

  bool initTrainInput(std::string const& rawDataPath, std::string const& targetPath);

private:
  DataPreprocessHelper(){};
};

#endif