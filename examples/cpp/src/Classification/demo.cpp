#include <chrono>
#include <algorithm>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../utils/Reader.h"
#include "../utils/Preprocessor.h"
#include "EngineWrapper.hpp"
#include "Launcher.hpp"
#include "TEEClsEngine.h"
#include "../utils/utils.h"

#define _NX_WIN_NAME_	"classify"
#define _NX_IMG_WIDTH_	224
#define _NX_IMG_HEIGHT_	224
#define _NX_IMG_SCALER_	256.0

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

// DEMO-specific Preprocessor
// This is an example of the pre-process of the input image, 
// It needs to be consistent with the preprocessing you use in training
class SimplePreprocessor : public Preprocessor {
    cv::Mat process(cv::Mat image) {
        if (image.empty())
            return image;

        cv::Mat mid(image.size(), image.type());
        cv::cvtColor(image, mid, cv::COLOR_BGR2RGB);
        cv::Mat img256;
        double scale = min(image.cols, image.rows) / _NX_IMG_SCALER_;
        int newHeight, newWidth;
        newHeight = (int)(image.rows / scale);
        newWidth = (int)(image.cols / scale);
        cv::resize(mid, img256, cv::Size(newWidth, newHeight));  //Resize to 256, keep the origin ratio

        int x, y;
        x = (newWidth - _NX_IMG_WIDTH_) / 2;
        y = (newHeight - _NX_IMG_HEIGHT_) / 2;
        cv::Mat img224(img256, cv::Rect(x, y, _NX_IMG_WIDTH_, _NX_IMG_HEIGHT_));  //centor crop to 224*224
        img224.copyTo(image);
		return image;
    }
};

// DEMO-specific Callback-Args for Engine
struct _TimeInfo {
	int duration;
	std::vector<std::string> vtLabel;  //labels
	std::chrono::high_resolution_clock::time_point bgnT;
	std::chrono::high_resolution_clock::time_point endT;
	_TimeInfo(std::string const lbl_filename) {
		duration = 0;
		_GetNameList2(lbl_filename, vtLabel);
	}
};

// DEMO-specific Callback for Engine
NXRet _ResultCB(nxvoid *pPrivateData, nxui8 const *retBuf, nxi32 bufLen, nxui64 id, nxi32 classNum) {
    static std::chrono::high_resolution_clock::time_point t1_ = std::chrono::high_resolution_clock::now();
    _TimeInfo *info = (_TimeInfo*)pPrivateData;
    if (info->duration++ == 0) {
        info->bgnT = t1_;
        info->endT = t1_;
    }

    std::chrono::high_resolution_clock::time_point cur = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> proc_time = std::chrono::duration<double>(cur - info->endT);
    info->endT = cur;

    nxreal32 fps = 0.0f;
    unsigned int maxi = 0;
    if (classNum > 0) {
        nxreal32 const *prob = (nxreal32 const *)(retBuf + _NX_IMG_WIDTH_ * _NX_IMG_HEIGHT_ * 3);
        for (int i = 1; i < classNum; ++i) {
            if (prob[maxi] < prob[i]) maxi = i;
        }
    }
	char strFPS[260] = { 0 };
	fps = info->duration * 1.0f / std::chrono::duration<double>(info->endT - info->bgnT).count();
    printf("%d: time: %.2fms\t %.2f FPS\t%s\n", info->duration, proc_time.count() * 1000.0, fps, info->vtLabel[maxi].c_str());

#ifndef NO_GUI
    sprintf(strFPS, "FPS: %.2f  ", fps);
    if (maxi < info->vtLabel.size()) strcat(strFPS, info->vtLabel[maxi].c_str());

    cv::Mat img(_NX_IMG_HEIGHT_, _NX_IMG_WIDTH_, CV_8UC3, (void*)retBuf);
    cv::Mat mbgr(img.size(), img.type());
    cv::cvtColor(img, mbgr, cv::COLOR_RGB2BGR);
    img = mbgr;
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    int thickness = 1;
    int baseline = 0;
    cv::Point orig(5, 215);
    float scales = 0.4;
    cv::Size text = cv::getTextSize(strFPS, fontface, scales, thickness, &baseline);
    cv::rectangle(img, orig + cv::Point(0, baseline), orig + cv::Point(text.width, -text.height), CV_RGB(50, 60, 70), CV_FILLED);
    cv::putText(img, strFPS, orig, fontface, scales, CV_RGB(255, 255, 0), thickness, 8);

    char winn[260] = {0};
    sprintf(winn, "%s_%d", _NX_WIN_NAME_, id & 0xffffffff);
    cv::imshow(winn, img);
    cv::waitKey(1);
#endif

    return TEE_RET_SUCCESS;
}

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
    NXEngineConf config;
	GenerateClassificationEngineConfigFromCmdArgs(&config, cmdArgs);
	
	// Callback Func & Args
	_TimeInfo timeInfo(cmdArgs[_NX_MODEL_PATH_] + cmdArgs[_NX_LABEL_NAME_]);
	config.pCBData = &timeInfo;
	config.pCB = _ResultCB;

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


