#ifndef _ENGINE_WRAPPER_H_
#define _ENGINE_WRAPPER_H_

#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "TEETypes.h"
#include "TEEErrorCode.h"
#include "TEEDetEngine.h"

#pragma warning (disable:4996)

// Function types for interface functions
typedef unsigned int(*LPTEEGetVersion)(void);

typedef int(*LPTEEDetCreateEngine)(IN TEEDetConfig const *pcConfig, OUT void **ppEngine);

typedef int(*LPTEEDetForward)(IN void *pEngine, IN TEEImage const *pImg, IN const int nOriginalWidth, IN const int nOriginalHeight, OUT TEERet *pRet);

typedef int(*LPTEEDetDestroyEngine)(IN OUT void **ppEngine);

class EngineWrapper {
    public:
        EngineWrapper(TEEDetConfig *config) {
            config_ = config;
            engine_ = 0;
#ifdef _WIN32
			hdll_ = 0;
			TEEGetVersion_ = 0;
			TEEDetCreateEngine_ = 0;
			TEEDetForward_ = 0;
			TEEDetDestroyEngine_ = 0;
#else
			TEEGetVersion_ = TEEGetVersion;
			TEEDetCreateEngine_ = TEEDetCreateEngine;
			TEEDetForward_ = TEEDetForward;
			TEEDetDestroyEngine_ = TEEDetDestroyEngine;
#endif
        }
        ~EngineWrapper() {
            if (engine_) {
				TEEDetDestroyEngine_(&engine_);
                engine_ = 0;
            }
#ifdef _WIN32
			if (hdll_) {
				FreeLibrary(hdll_);
				hdll_ = 0;
			}
#endif
			TEEGetVersion_ = 0;
			TEEDetCreateEngine_ = 0;
			TEEDetForward_ = 0;
			TEEDetDestroyEngine_ = 0;

			config_ = 0;
        }
        bool create() {
#ifdef _WIN32
			char path[512] = { 0 };
			GetModuleFileName(0, path, 512);
			char *tail = path + strlen(path);
			while (tail != path) {
				if (*tail == '\\' || *tail == '/') break;
				else --tail;
			}
			*tail = 0;
			strcat(path, "\\TEEDetectorFRCNN.dll");
			// printf("%s\n", path);
			hdll_ = LoadLibrary(path);
			if (!hdll_) {
				printf("TEEDetectorFasterRcnn.dll not found. exit.\n");
				hdll_ = 0;
				return false;
			}
			TEEGetVersion_ = (LPTEEGetVersion)GetProcAddress(hdll_, "TEEGetVersion");
			TEEDetCreateEngine_ = (LPTEEDetCreateEngine)GetProcAddress(hdll_, "TEEDetCreateEngine");
			TEEDetForward_ = (LPTEEDetForward)GetProcAddress(hdll_, "TEEDetForward");
			TEEDetDestroyEngine_ = (LPTEEDetDestroyEngine)GetProcAddress(hdll_, "TEEDetDestroyEngine");
			if (!TEEGetVersion_ || !TEEDetCreateEngine_ || !TEEDetForward_ || !TEEDetDestroyEngine_) {
				printf("get dll interface fail. exit\n");
			}
#endif
			int iRet = TEEDetCreateEngine_(config_ , &engine_ );

            if (iRet == TEE_RET_SUCCESS && engine_) {
                return true;
            } else {
				printf("Create Engine Failed: %d\n", iRet);
                engine_ = 0;
                return false;
            }
        }
		NXRet push(TEEImage *img, const int nOriginalWidth, const int nOriginalHeight, TEERet *pRet) {
            return TEEDetForward_(engine_, img,nOriginalWidth,nOriginalHeight, pRet);
        }
		void clear() {
			TEEDetDestroyEngine_(&engine_);
		}

    private:
		TEEDetConfig *config_;
        nxvoid *engine_;

#ifdef _WIN32
		HMODULE hdll_;
#endif
		LPTEEGetVersion  TEEGetVersion_;
		LPTEEDetCreateEngine TEEDetCreateEngine_;
		LPTEEDetForward TEEDetForward_;
		LPTEEDetDestroyEngine TEEDetDestroyEngine_;

};

#endif  // _ENGINE_WRAPPER_H_
