#ifndef KMEANS_MOSAIC_HPP
#define KMEANS_MOSAIC_HPP

#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include "statistical_features.hpp"

cv::Mat generateMosaicWithKMeans(
    const cv::Mat& inputImage,
    std::map<std::string, StatisticalFeatures>& datasetStats,
    int blockSize,
    GenerateMosaicParams params,
    int k
);

#endif // KMEANS_MOSAIC_HPP