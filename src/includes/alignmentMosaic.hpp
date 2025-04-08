#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include <future>
#include <mutex>

namespace fs = std::filesystem;

/**
 * @brief Generate a mosaic using the alignment method, multi-threaded
 * 
 * @param inputImage 
 * @param blockSize 
 * @param folderPath 
 * @return cv::Mat 
 */
cv::Mat generateMosaicUsingAlignment(const cv::Mat& inputImage, int blockSize, const std::string& folderPath);

/**
 * @brief Generate a mosaic using the alignment method, single-threaded
 * 
 * @param inputImage 
 * @param blockSize 
 * @param folderPath 
 * @param uniquesImagettes 
 * @return cv::Mat 
 */
cv::Mat generateMosaicUsingAlignmentSingleThread(const cv::Mat& inputImage, int blockSize, const std::string& folderPath, bool uniquesImagettes);
