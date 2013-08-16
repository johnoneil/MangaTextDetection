/*
 # vim: set ts=2 expandtab:

FILE: StrokeWidthTranform.hpp
AUTHOR: John O'Neil
DATE: Sunday, July 8th 2013
EMAIL: oneil.john@gmail.com
DESC: Implementation of stroke width tranform algorithm as described
in http://www.math.tau.ac.il/~turkel/imagepapers/text_detection.pdf

This uses the opencv2 C++ api

*/
#ifndef __STROKEWIDTHTRANSFORM_HPP__
#define __STROKEWIDTHTRANSFORM_HPP__

#include <opencv2/core/core.hpp>

cv::Mat StrokeWidthTransform(const cv::Mat& image);

void EstimatePixelStrokeWidth(const int row,const int col,const cv::Mat& edges,
  const cv::Mat& gradient_x,const cv::Mat& gradient_y, cv::Mat& output);


#endif
