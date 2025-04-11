#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include <FL/Fl.H>
#include <FL/Fl_Progress.H>
#include "../statistical_features.hpp"

#define STATISTICAL_FEATURES_FILE "stats_features.txt"
namespace fs = std::filesystem;

/**
 * @brief Prétraitement du dataset pour obtenir les statistiques des images
 * 
 * @param folderPath 
 * @return std::map<std::string, StatisticalFeatures> 
 */
std::map<std::string, StatisticalFeatures> preprocessDatasetStats(const std::string& folderPath);

/**
 * @brief Précalcul des statistiques des images du dataset si elles n'ont pas déjà été calculées
 * 
 * @param folderPath 
 * @return std::map<std::string, StatisticalFeatures> 
 */
std::map<std::string, StatisticalFeatures> checkIfAlreadyPreProcessed(const std::string& folderPath);

/**
 * @brief Calcul de la distance entre deux images
 * 
 * @param a 
 * @param b 
 * @param params 
 * @return double 
 */
double computeDistance(StatisticalFeatures a, StatisticalFeatures b, GenerateMosaicParams params);

/**
 * @brief Découpe une image en blocs de taille blockSize
 * 
 * @param image 
 * @param blockSize 
 * @return std::vector<cv::Mat> 
 */
std::vector<cv::Mat> splitImageIntoBlocks(const cv::Mat& image, int blockSize);

/**
 * @brief Génère une mosaïque à partir d'une image et d'un ensemble d'images de référence
 * 
 * @param inputImage 
 * @param meanValues 
 * @param blockSize 
 * @param params 
 * @return cv::Mat 
 */
cv::Mat generateMosaic(const cv::Mat& inputImage, std::map<std::string, StatisticalFeatures> &meanValues, int blockSize, GenerateMosaicParams params, Fl_Progress* progressBar);
/**
 * @brief Calcul du PSNR entre deux images
 * 
 * @param I1 
 * @param I2 
 * @return float 
 */
float PSNR(const cv::Mat& I1, const cv::Mat& I2);

cv::Mat fitBlocks(const cv::Mat& img, int blockSize);