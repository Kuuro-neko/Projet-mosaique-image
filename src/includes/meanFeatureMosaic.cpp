#include "../includes/meanFeatureMosaic.hpp"

StatisticalFeatures processImageStats(const cv::Mat& img){
    // Color
    cv::Scalar mean = cv::mean(img);
    Color c = {mean[2], mean[1], mean[0]};

    // Variance & Energy
    cv::Scalar variance;
    cv::Scalar meanSquares;
    cv::meanStdDev(img, meanSquares, variance);
    Variance v = {variance[2], variance[1], variance[0]};
    Energy e = {meanSquares[2], meanSquares[1], meanSquares[0]};

    // Skewness
    cv::Scalar meanCubed;
    cv::Scalar meanCubedTimesMean;
    cv::meanStdDev(img.mul(img.mul(img)), meanCubed, variance);
    cv::meanStdDev(img.mul(img.mul(img.mul(img))), meanCubedTimesMean, variance);
    Skewness s = {meanCubed[2] - 3 * mean[2] * meanSquares[2] + 2 * mean[2] * mean[2] * mean[2],
                    meanCubed[1] - 3 * mean[1] * meanSquares[1] + 2 * mean[1] * mean[1] * mean[1],
                    meanCubed[0] - 3 * mean[0] * meanSquares[0] + 2 * mean[0] * mean[0] * mean[0]};
    
    return {c, v, s, e};
}

std::map<std::string, StatisticalFeatures> preprocessDatasetStats(const std::string& folderPath){
    std::map<std::string, StatisticalFeatures> meanValues;

    int progress = 0;
    int total = std::distance(fs::directory_iterator(folderPath), fs::directory_iterator{});

    for(const auto& entry : fs::directory_iterator(folderPath)){
        try {
            cv::Mat img = cv::imread(entry.path().string());
            
            meanValues[entry.path().string()] = processImageStats(img);

            progress++;
            if (progress % 250 == 0){
                std::cout << "Progress : " << int(progress / (float)total * 100) << "%" << std::flush << "\r";
            }
        } catch (cv::Exception& e){
            // There is probably a json file in the dataset folder, catching it's error here
            std::cout << std::endl;
            std::cout << "Error while processing image : " << entry.path().string() << std::endl;
        }
    }
    std::cout << "Progress : 100%" << std::endl;
    return meanValues;
}

double computeDistance(StatisticalFeatures a, StatisticalFeatures b, GenerateMosaicParams params){
    double distance = 0;
    if(params.meanColor){
        distance += sqrt(pow(a.mean.r - b.mean.r, 2) + pow(a.mean.g - b.mean.g, 2) + pow(a.mean.b - b.mean.b, 2));
    }
    if(params.variance){
        distance += sqrt(pow(a.variance.r - b.variance.r, 2) + pow(a.variance.g - b.variance.g, 2) + pow(a.variance.b - b.variance.b, 2));
    }
    if(params.skewness){
        distance += sqrt(pow(a.skewness.r - b.skewness.r, 2) + pow(a.skewness.g - b.skewness.g, 2) + pow(a.skewness.b - b.skewness.b, 2));
    }
    if(params.energy){
        distance += sqrt(pow(a.energy.r - b.energy.r, 2) + pow(a.energy.g - b.energy.g, 2) + pow(a.energy.b - b.energy.b, 2));
    }
    return distance;
}