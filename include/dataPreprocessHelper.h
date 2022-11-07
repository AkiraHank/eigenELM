#ifndef DATAPREPROCESSHELPER_H
#define DATAPREPROCESSHELPER_H

#include <string>
#include "fileUtils.hpp"

class dataPreprocessHelper {
public:
  dataPreprocessHelper& getInstance() {
    static dataPreprocessHelper instance;
    return instance; 
  }

  bool initTrainInput(std::string const& rawDataPath, std::string const& targetPath);

private:
  dataPreprocessHelper();
};

#endif