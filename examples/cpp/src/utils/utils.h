#include <map>
#include <string>
#include <vector>
#include "TEEClsEngine.h"
#include "TEEDetEngine.h"

#define _NX_STICK_NUM_  "stickNum"
#define _NX_THREAD_NUM_ "threadNum"
#define _NX_MODEL_PATH_ "modelPath"
#define _NX_STICK_CNN_  "stickCNN"
#define _NX_HOST_NET_  "hostNet"
#define _NX_HOST_NET_PROTO_   "hostNetProto"
#define _NX_HOST_NET_WEIGHTS_   "hostNetWeights"
#define _NX_STICK_USER_INPUT_ "stickUserInput"
#define _NX_NEG_THRE_FASTER_RCNN_ "negThreFasterRcnn"
#define _NX_LABEL_NAME_ "labelName"
#define _NX_NET_TYPE_   "netType"
#define _NX_ClASS_NUM_  "classNum"
#define _NX_FILELIST_	"fileList"
#define _NX_SG_BEGIN_ID_       "sgBeginID"
#define _NX_DELAY_TIME_       "delayTime"

void printfUsage(void);

std::map<std::string, std::string> ParseArgs(int argc, char *argv[]);

void GenerateClassificationEngineConfigFromCmdArgs(NXEngineConf *config, std::map<std::string, std::string> &cmdArgs);

void GenerateDetectionEngineConfigFromCmdArgs(TEEDetConfig *config, std::map<std::string, std::string> &cmdArgs);

void _GetNameList2(std::string const &name, std::vector<std::string> &vtName);
void _GetNameList(std::string const &name, std::vector<std::string> &vtName);






