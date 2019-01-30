#ifndef _PREPROCESSOR_H_
#define _PREPROCESSOR_H_

#include "TEEClsEngine.h"

class Preprocessor {
public:
	virtual cv::Mat process(cv::Mat input) = 0;
};

#endif  // _PREPROCESSOR_H_




