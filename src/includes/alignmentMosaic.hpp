#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

/**
 * @brief Compute the alignment score between two strings
 * 
 * @param a 
 * @param b 
 * @param match 
 * @param mismatch 
 * @param gap 
 * @return int 
 */
int alignmentScore(const std::string &a, const std::string &b, int match = 1, int mismatch = -1, int gap = -1);

/**
 * @brief Get the image as a string
 * 
 * @param img 
 * @return std::string 
 */
std::string getImageAsString(const cv::Mat& img);

/**
 * @brief Generate a resized subset of the dataset
 * 
 * @param folderPath 
 * @param blocSize 
 * @param n 
 * @return std::string 
 */
std::string generateResizedDataset(const std::string& folderPath, int blocSize, int n = -1);

/**
 * @brief Generate a mosaic using the alignment method
 * 
 * @param inputImage 
 * @param blockSize 
 * @param folderPath 
 * @param uniquesImagettes 
 * @return cv::Mat 
 */
cv::Mat generateMosaicUsingAlignment(const cv::Mat& inputImage, int blockSize, const std::string& folderPath, bool uniquesImagettes = false);
