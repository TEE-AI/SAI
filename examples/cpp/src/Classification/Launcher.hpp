#ifndef _LAUNCHER_H_
#define _LAUNCHER_H_

#include "../utils/Reader.h"
#include "../utils/Preprocessor.h"
#include "EngineWrapper.hpp"

class Launcher {
    public:
        Launcher(Reader *reader, Preprocessor *processor, EngineWrapper *engine) {
            reader_ = reader;
            engine_ = engine;
            if (processor)
                reader_->register_processor(processor);
        }
        ~Launcher() {
			reader_ = 0;
			engine_ = 0;
        }
        void run() {
            nxui64 id;
            cv::Mat img = reader_->get();
            while (!img.empty()) {
				NXImg param;
				param.data = img.data;
				param.h = img.rows;
				param.w = img.cols;
				param.pixfmt = ePixfmtRGB;
				NXRet status = engine_->push(&param, &id);
				if (status != TEE_RET_SUCCESS) {
					// Error
					printf("Push task. id: %lld ret: %d\n", id, status);
				}
                img = reader_->get();
            }
			engine_->clear();
        }

    private:
        Reader *reader_;
        EngineWrapper *engine_;
};

#endif  // _LAUNCHER_H_


