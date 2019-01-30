#ifndef _READER_H_
#define _READER_H_

//#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include "Preprocessor.h"

class Reader {
    public:
        Reader(std::vector<std::string> *filelist);
        ~Reader();
        void register_processor(Preprocessor *processor);
        virtual cv::Mat get() = 0;
    protected:
        std::vector<std::string> *filelist_;
        int fileIndex_;
        Preprocessor *processor_;
};

/* This an example of the ImageReader which can process the image file list
*  You can re-implement the ImageReader based on your requirements
*/
class ImageReader : public Reader {
    public:
        ImageReader(std::vector<std::string> *filelist);
		cv::Mat get();
};

/* This an example of the VideoReader which can process the video file list
*  You can re-implement the VideoReader based on your requirements
*/
/*
class VideoReader : public Reader {
public:
	VideoReader(std::vector<std::string> *filelist);
	cv::Mat get();
private:
	cv::VideoCapture videoCap_;
};
*/

#endif  // _READER_H_




