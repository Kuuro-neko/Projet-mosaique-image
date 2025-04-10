#ifndef STATISTICAL_FEATURES_HPP
#define STATISTICAL_FEATURES_HPP

#include <string>
#include <opencv2/opencv.hpp>

struct Color {
    double r;
    double g;
    double b;
};

struct Variance {
    double r;
    double g;
    double b;
};

struct Skewness {
    double r;
    double g;
    double b;
};

struct Energy {
    double r;
    double g;
    double b;
};

struct StatisticalFeatures {
    Color mean;
    Variance variance;
    Skewness skewness;
    Energy energy;
};

struct Tamura {
    double coarseness;
    double contrast;
    double directionality;
};

struct GenerateMosaicParams {
    bool meanColor = true;
    bool variance = true;
    bool skewness = false;
    bool energy = false;
    bool reuseImages = false;

    GenerateMosaicParams(bool meanColor = true, bool variance = true, bool skewness = false, bool energy = false, bool reuseImages = false);

    void setFromBitArray(const std::string& bitArray);
    std::string toString() const;
};

StatisticalFeatures processImageStats(const cv::Mat& img);

#endif // STATISTICAL_FEATURES_HPP
