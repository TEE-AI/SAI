#ifndef _ENGINE_WRAPPER_H_
#define _ENGINE_WRAPPER_H_

#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "NXErrorCode.h"
#include "INXInferenceEngine.h"

// Function types for interface functions
typedef NXRet(*LPNXCreateInferenceEngine)(nxvoid **ppEngine, NXEngineConf const *pConf);

typedef NXRet(*LPNXPushTask)(nxvoid *engine, NXImg const *pImg, nxui64 *pID);

typedef NXRet(*LPNXClearAllTask)(nxvoid *engine);

typedef NXRet(*LPNXDestroyInferenceEngine)(nxvoid *pEngine);

class EngineWrapper {
    public:
        EngineWrapper(NXEngineConf *config) {
            config_ = config;
            engine_ = 0;
#ifdef _WIN32
			hdll_ = 0;
			NXCreateInferenceEngine_ = 0;
			NXPushTask_ = 0;
			NXClearAllTask_ = 0;
			NXDestroyInferenceEngine_ = 0;
#else
			NXCreateInferenceEngine_ = NXCreateInferenceEngine;
			NXPushTask_ = NXPushTask;
			NXClearAllTask_ = NXClearAllTask;
			NXDestroyInferenceEngine_ = NXDestroyInferenceEngine;
#endif
        }
        ~EngineWrapper() {
            if (engine_) {
                NXDestroyInferenceEngine_(engine_);
                engine_ = 0;
            }
#ifdef _WIN32
			if (hdll_) {
				FreeLibrary(hdll_);
				hdll_ = 0;
			}
#endif
			NXCreateInferenceEngine_ = 0;
			NXPushTask_ = 0;
			NXClearAllTask_ = 0;
			NXDestroyInferenceEngine_ = 0;

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
			strcat(path, "/TEEClassifier.dll");
			// printf("%s\n", path);
			hdll_ = LoadLibrary(path);
			if (!hdll_) {
				printf("TEEClassifier.dll not found. exit.\n");
				hdll_ = 0;
				return false;
			}
			NXCreateInferenceEngine_ = (LPNXCreateInferenceEngine)GetProcAddress(hdll_, "NXCreateInferenceEngine");
			NXPushTask_ = (LPNXPushTask)GetProcAddress(hdll_, "NXPushTask");
			NXClearAllTask_ =(LPNXClearAllTask)GetProcAddress(hdll_, "NXClearAllTask");
			NXDestroyInferenceEngine_ = (LPNXDestroyInferenceEngine)GetProcAddress(hdll_, "NXDestroyInferenceEngine");
			if (!NXCreateInferenceEngine_ || !NXPushTask_ || !NXClearAllTask_ || !NXDestroyInferenceEngine_) {
				printf("get dll interface fail. exit\n");
				return false;
			}
#endif
            NXRet status = NXCreateInferenceEngine_(&engine_, config_);

            if (status == NX_RET_SUCCESS && engine_) {
                return true;
            } else {
				printf("Create Engine Failed: %d\n", status);
                engine_ = 0;
                return false;
            }
        }
		NXRet push(NXImg *img, nxui64 *id) {
            return NXPushTask_(engine_, img, id);
        }
		void clear() {
			NXClearAllTask_(engine_);
		}

    private:
        NXEngineConf *config_;
        nxvoid *engine_;

#ifdef _WIN32
		HMODULE hdll_;
#endif
		LPNXCreateInferenceEngine NXCreateInferenceEngine_;
		LPNXPushTask NXPushTask_;
		LPNXClearAllTask NXClearAllTask_;
		LPNXDestroyInferenceEngine NXDestroyInferenceEngine_;

};

#endif  // _ENGINE_WRAPPER_H_
