#include "Reader.h"

Reader::Reader(std::vector<std::string> *filelist) {
    filelist_ = filelist;
    fileIndex_ = 0;
    processor_ = 0;
}

Reader::~Reader() {
    filelist_ = 0;
    fileIndex_ = 0;
    processor_ = 0;
}

void Reader::register_processor(Preprocessor *processor) {
    processor_ = processor;
}

ImageReader::ImageReader(std::vector<std::string> *filelist) : Reader(filelist) {}

cv::Mat ImageReader::get() {
    if (fileIndex_ < filelist_->size()) {
		cv::Mat cvImage= cv::imread((*filelist_)[fileIndex_++]);
        if (processor_) {
            cv::Mat procImage = processor_->process(cvImage);
			return procImage;
		}
		else {
			return cvImage;
		}
	}
	else {
		return cv::Mat();
	}
}

/*
VideoReader::VideoReader(std::vector<std::string> *filelist) : Reader(filelist) {
    // Open the first file if possible
    if (fileIndex_ < filelist_->size()) {
        videoCap_.open((*filelist_)[fileIndex_]);
    }
}

cv::Mat VideoReader::get() {
    cv::Mat rawImage;
    videoCap_.read(rawImage);
    if (rawImage.empty()) {
        // Reach the end of the current file
        fileIndex_++;
        if (fileIndex_ < filelist_->size()) {
            videoCap_.open((*filelist_)[fileIndex_]);
            videoCap_.read(rawImage);
        }
    }
    if (processor_) {
		cv::Mat procImage = processor_->process(rawImage);
		return procImage;
    }
    return rawImage;
}
*/



