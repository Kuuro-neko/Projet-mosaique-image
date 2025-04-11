#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

struct Color{
    double r;
    double g;
    double b;
};

struct Variance{
    double r;
    double g;
    double b;
};

struct Skewness{
    double r;
    double g;
    double b;
};

struct Energy {
    double r;
    double g;
    double b;
};

struct StatisticalFeatures{
    Color mean;
    Variance variance;
    Skewness skewness;
    Energy energy;
};

/**
 * @brief Parameters for the generation of the mosaic
 * 
 */
struct GenerateMosaicParams{
    bool meanColor = true;
    bool variance = true;
    bool skewness = false;
    bool energy = false;
    bool reuseImages = false;
    
    /**
     * @brief Construct a new Generate Mosaic Params object, default constructor
     * 
     * @param meanColor 
     * @param variance 
     * @param skewness 
     * @param energy 
     * @param reuseImages 
     */
    GenerateMosaicParams(bool meanColor = true, bool variance = true, bool skewness = false, bool energy = false, bool reuseImages = false) : meanColor(meanColor), variance(variance), skewness(skewness), energy(energy), reuseImages(reuseImages) {}
    
    void setFromBitArray(const std::string& bitArray){
        meanColor = bitArray[0] == '1';
        variance = bitArray[1] == '1';
        skewness = bitArray[2] == '1';
        energy = bitArray[3] == '1';
        reuseImages = bitArray[4] == '1';
    }

    std::string toString() const {
        std::string result = "meanColor : " + std::to_string(meanColor) + ", ";
        result += "variance : " + std::to_string(variance) + ", ";
        result += "skewness : " + std::to_string(skewness) + ", ";
        result += "energy : " + std::to_string(energy) + ", ";
        result += "reuseImages : " + std::to_string(reuseImages);
        return result;
    }
};


/**
 * @brief Extraction des statistiques d'une image
 * 
 * @param img 
 * @return StatisticalFeatures 
 */
StatisticalFeatures processImageStats(const cv::Mat& img);

/**
 * @brief Prétraitement du dataset pour obtenir les statistiques des images
 * 
 * @param folderPath 
 * @return std::map<std::string, StatisticalFeatures> 
 */
std::map<std::string, StatisticalFeatures> preprocessDatasetStats(const std::string& folderPath);

/**
 * @brief Calcul de la distance entre deux images
 * 
 * @param a 
 * @param b 
 * @param params 
 * @return double 
 */
double computeDistance(StatisticalFeatures a, StatisticalFeatures b, GenerateMosaicParams params);