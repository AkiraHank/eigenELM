#include <filesystem>
#include <string>
namespace fs = std::filesystem;

namespace fileUtils {

bool checkInputFileValid(std::string const& path) {
  return fs::exists(path);
}

bool checkOutputFileValid(std::string const& path) {
  int index = path.find_last_of('/');
  if (index != std::string::npos) {
    std::string dir = path.substr(0, index);
    if (!fs::exists(dir)) {
      return fs::create_directory(dir);
    }
  }
  return true;
}

}
