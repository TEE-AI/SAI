#include <chrono>
#include <algorithm>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../utils/Reader.h"
#include "../utils/Preprocessor.h"
#include "EngineWrapper.hpp"
#include "Launcher.hpp"
#include "TEEDetEngine.h"
#include "../utils/utils.h"

#pragma warning (disable:4996)

#define _NX_WIN_NAME_	"Detection_FasterRcnn"
#define _NX_IMG_WIDTH_	224
#define _NX_IMG_HEIGHT_	224
#define _NX_IMG_SCALER_	256.0

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

// DEMO-specific Preprocessor
// This is an example of the pre-process of the input image, 
// It needs to be consistent with the preprocessing you used in training
class SimplePreprocessor : public Preprocessor {
    cv::Mat process(cv::Mat image) {
		return image;
    }
};

#define _NX_MAX_PATH_LEN_   512

int main(int argc, char *argv[]) {
    // Parse cmd-line args
    if (argc < 9) {
        printfUsage();
        return 1;
    }
    std::map<std::string, std::string> cmdArgs = ParseArgs(argc, argv);
    if (cmdArgs.empty()) {
        // Error
        return 1;
    }

	// Create Preprocessor
	Preprocessor *processor = new SimplePreprocessor();

    // Create Image Reader
	std::vector<std::string> filelist;
	_GetNameList(cmdArgs[_NX_MODEL_PATH_] + cmdArgs[_NX_FILELIST_], filelist);
    Reader *reader = new ImageReader(&filelist);

    // Create EngineConfig
	TEEDetConfig config;
	GenerateDetectionEngineConfigFromCmdArgs(&config, cmdArgs);
	
	// Create Engine
    EngineWrapper *engine = new EngineWrapper(&config);
	if (!engine->create()) {
		// Error
		printf("Create Engine Failed!\n");
		return 1;
	}

    // Launch the task
    Launcher launcher(reader, processor, engine);
    launcher.run();
	
	// Clear Resources
	delete engine;
	delete reader;
	delete processor;

	return 0;
}


