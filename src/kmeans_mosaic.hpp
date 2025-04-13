#ifndef KMEANS_MOSAIC_HPP
#define KMEANS_MOSAIC_HPP

#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include "statistical_features.hpp"

#include <FL/Fl.H>
#include <FL/Fl_Progress.H>

cv::Mat generateMosaicWithKMeans(
    const cv::Mat& inputImage,
    std::map<std::string, StatisticalFeatures>& datasetStats,
    int blockSize,
    GenerateMosaicParams params,
    int k,
    Fl_Progress* progressBar
);

#endif // KMEANS_MOSAIC_HPP