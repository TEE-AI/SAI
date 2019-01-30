#ifndef _LAUNCHER_H_
#define _LAUNCHER_H_

#include "../utils/Reader.h"
#include "../utils/Preprocessor.h"
#include "EngineWrapper.hpp"

#define _NX_WIN_NAME_	"Detection_FasterRcnn"


void _draw_preson(cv::Mat &img, TEERet &detRet) {
	{
		cv::Scalar vtColor0 = cv::Scalar(255, 0, 0); // b
		cv::Scalar vtColor1 = cv::Scalar(0, 255, 0); // g
		cv::Scalar vtColor2 = cv::Scalar(0, 0, 255); // r
		char *strlables[3] = { "BG", "Helmet", "Person" };
		for (size_t obj = 0; obj < detRet.num; obj++) {
			{
				cv::rectangle(img, cv::Rect(detRet.vtDetData[obj].x, detRet.vtDetData[obj].y, detRet.vtDetData[obj].w, detRet.vtDetData[obj].h), (detRet.vtDetData[obj].eType == eTEEDetHead ? vtColor2 : vtColor1), 3);
			}
		}
	}
}

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
				cv::Mat inputMat(224, 224, img.type());
				cv::resize(img, inputMat, cv::Size(224, 224));
				TEEImage param;
				param.data = inputMat.data;
				param.fmt = eTeeImgFmtBGR;
				param.w = inputMat.cols;
				param.h = inputMat.rows;
				int nOriginalWidth = img.cols;
				int nOriginalHeight = img.rows;
				TEERet detRet;
				detRet.num = 0;
				detRet.vtDetData = 0;
				NXRet status = engine_->push(&param, nOriginalWidth, nOriginalHeight, &detRet);
				if (status != TEE_RET_SUCCESS) {
					// Error
					printf("Push task ret: %d\n", status);
				}
				_draw_preson(img, detRet);
				cv::Mat outputMat(1080, 720, img.type());
				cv::resize(img, outputMat, cv::Size(1080, 720));
				cv::imshow(_NX_WIN_NAME_, outputMat);
				cv::waitKey(100);
                img = reader_->get();
            }
			engine_->clear();
        }

    private:
        Reader *reader_;
        EngineWrapper *engine_;
};

#endif  // _LAUNCHER_H_


