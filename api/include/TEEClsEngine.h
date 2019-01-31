/*
*
* Copyright (c) 2018-2019 TEE.COM. All Rights Reserved
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#ifndef _TEE_CLASSIFY_ENGINE_H_
#define _TEE_CLASSIFY_ENGINE_H_
#ifdef __cplusplus
extern "C" {
#endif
#ifdef _WIN32
    #define NXDLL __declspec(dllimport)
#else
    #define NXDLL 
#endif
    typedef enum {
        ePixfmtBGR = 0, // BGRBGR
        ePixfmtRGB = 1, // RGBRGB
    }TEEPixFmt;

    typedef struct {
        int w;
        int h;
        TEEPixFmt pixfmt;
        unsigned char *data;
    }TEEImg;

    /* Inference engine send result data callback function */
    typedef int (*LPResultCB)(void *pPrivateData, unsigned char const *retBuf, int bufLen, unsigned long long id, int classNum);

    typedef struct {
		char const* confJsonData; //json format data buffer
        LPResultCB pCB;
        void *pCBData;
    }TEEClsConf;

    /*  Create inference engine */
	//int NXDLL NXCreateInferenceEngine(void **ppEngine, TEEClsConf const *pConf);
	int NXDLL TEEClsCreateEngine(void **ppEngine, TEEClsConf const *pConf);

    /* send an image to engine and engine set *pID value. engine will send the id to callback function */
	int NXDLL TEEClsPushTask(void *engine, TEEImg const *pImg, unsigned long long *pID);

    /* clear all task */
	int NXDLL TEEClsClearAllTask(void *engine);

	int NXDLL TEEClsDestroyEngine(void *pEngine);
#ifdef __cplusplus
}
#endif
#endif // _TEE_CLASSIFY_ENGINE_H_
