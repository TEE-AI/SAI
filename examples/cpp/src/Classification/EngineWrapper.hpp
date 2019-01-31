#ifndef _ENGINE_WRAPPER_H_
#define _ENGINE_WRAPPER_H_

#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "TEEErrorCode.h"
#include "TEEClsEngine.h"

// Function types for interface functions
typedef int(*LPTEEClsCreateEngine)(void **ppEngine, TEEClsConf const *pConf);

typedef int(*LPTEEClsPushTask)(void *engine, TEEImg const *pImg, unsigned long long *pID);

typedef int(*LPTEEClsClearAllTask)(void *engine);

typedef int(*LPTEEClsDestroyEngine)(void *pEngine);

class EngineWrapper {
    public:
        EngineWrapper(TEEClsConf *config) {
            config_ = config;
            engine_ = 0;
#ifdef _WIN32
			hdll_ = 0;
			TEEClsCreateEngine_ = 0;
			TEEClsPushTask_ = 0;
			TEEClsClearAllTask_ = 0;
			TEEClsDestroyEngine_ = 0;
#else
			TEEClsCreateEngine_ = TEEClsCreateEngine;
			TEEClsPushTask_ = TEEClsPushTask;
			TEEClsClearAllTask_ = TEEClsClearAllTask;
			TEEClsDestroyEngine_ = TEEClsDestroyEngine;
#endif
        }
        ~EngineWrapper() {
            if (engine_) {
                TEEClsDestroyEngine_(engine_);
                engine_ = 0;
            }
#ifdef _WIN32
			if (hdll_) {
				FreeLibrary(hdll_);
				hdll_ = 0;
			}
#endif
			TEEClsCreateEngine_ = 0;
			TEEClsPushTask_ = 0;
			TEEClsClearAllTask_ = 0;
			TEEClsDestroyEngine_ = 0;

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
			TEEClsCreateEngine_ = (LPTEEClsCreateEngine)GetProcAddress(hdll_, "TEEClsCreateEngine");
			TEEClsPushTask_ = (LPTEEClsPushTask)GetProcAddress(hdll_, "TEEClsPushTask");
			TEEClsClearAllTask_ =(LPTEEClsClearAllTask)GetProcAddress(hdll_, "TEEClsClearAllTask");
			TEEClsDestroyEngine_ = (LPTEEClsDestroyEngine)GetProcAddress(hdll_, "TEEClsDestroyEngine");
			if (!TEEClsCreateEngine_ || !TEEClsPushTask_ || !TEEClsClearAllTask_ || !TEEClsDestroyEngine_) {
				printf("get dll interface fail. exit\n");
				return false;
			}
#endif
            int status = TEEClsCreateEngine_(&engine_, config_);

            if (status == TEE_RET_SUCCESS && engine_) {
                return true;
            } else {
				printf("Create Engine Failed: %d\n", status);
                engine_ = 0;
                return false;
            }
        }
		int push(TEEImg *img, unsigned long long *id) {
            return TEEClsPushTask_(engine_, img, id);
        }
		void clear() {
			TEEClsClearAllTask_(engine_);
		}

    private:
		TEEClsConf *config_;
        void *engine_;

#ifdef _WIN32
		HMODULE hdll_;
#endif
		LPTEEClsCreateEngine TEEClsCreateEngine_;
		LPTEEClsPushTask TEEClsPushTask_;
		LPTEEClsClearAllTask TEEClsClearAllTask_;
		LPTEEClsDestroyEngine TEEClsDestroyEngine_;

};

#endif  // _ENGINE_WRAPPER_H_
