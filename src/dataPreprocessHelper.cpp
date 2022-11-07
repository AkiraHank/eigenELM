#include "dataPreprocessHelper.h"

bool dataPreprocessHelper::initTrainInput(std::string const& rawDataPath, std::string const& targetPath) {
  if (fileUtils::checkInputFileValid(rawDataPath)) {

  } else {
    return false;
  }
}
