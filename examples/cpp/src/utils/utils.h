#include <map>
#include <string>
#include <vector>
#include <sstream>

#include "TEEClsEngine.h"
#include "TEEDetEngine.h"

#define _TEE_JSON_CONFIG_ "configFile"
#define _TEE_DATA_TEST_PATH_ "dataTestPath"
#define _TEE_LABEL_NAME_ "labelName"
#define _TEE_FILE_LIST_	"fileList"

void printfUsage(void);

std::map<std::string, std::string> ParseArgs(int argc, char *argv[]);

void _GetNameList2(std::string const &name, std::vector<std::string> &vtName);
void _GetNameList(std::string const &name, std::vector<std::string> &vtName);

std::string readFileIntoString(const char * filename);





