#include "kmeans_mosaic.hpp"
#include <thread>
#include <set>
#include <iostream>
#include <atomic>
#include <cmath>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

/**
 * @brief Build a feature vector from image statistics
 * 
 * @param stats
 * @param params
 * @return std::vector<float>
 */
std::vector<float> featuresFromStat(const StatisticalFeatures& stats, const GenerateMosaicParams& params) {
    std::vector<float> features;
    if(params.meanColor){
        features.push_back(stats.mean.r);
        features.push_back(stats.mean.g);
        features.push_back(stats.mean.b);
    }
    if(params.variance){
        features.push_back(stats.variance.r);
        features.push_back(stats.variance.g);
        features.push_back(stats.variance.b);
    }
    if(params.skewness){
        features.push_back(stats.skewness.r);
        features.push_back(stats.skewness.g);
        features.push_back(stats.skewness.b);
    }
    if(params.energy){
        features.push_back(stats.energy.r);
        features.push_back(stats.energy.g);
        features.push_back(stats.energy.b);
    }
    return features;
}

/**
 * @brief Build a feature matrix from a map of image statistics
 * 
 * @param statsMap
 * @param params
 * @param filenamesOut
 * @return cv::Mat
 */
cv::Mat buildFeatureMatrix(const std::map<std::string, StatisticalFeatures>& statsMap, const GenerateMosaicParams& params, std::vector<std::string>& filenamesOut) {
    std::vector<std::vector<float>> featureVectors;
    filenamesOut.clear();

    for (const auto& [filename, stats] : statsMap) {
        featureVectors.push_back(featuresFromStat(stats, params));
        filenamesOut.push_back(filename);
    }

    cv::Mat featureMat(featureVectors.size(), featureVectors[0].size(), CV_32F);
    for (int i = 0; i < featureVectors.size(); ++i) {
        for (int j = 0; j < featureVectors[i].size(); ++j) {
            featureMat.at<float>(i, j) = featureVectors[i][j];
        }
    }

    return featureMat;
}

/**
 * @brief Generate a mosaic using k-means clustering
 * 
 * @param inputImage
 * @param datasetStats
 * @param blockSize
 * @return cv::Mat
 */
cv::Mat generateMosaicWithKMeans(
    const cv::Mat& inputImage,
    std::map<std::string, StatisticalFeatures>& datasetStats,
    int blockSize,
    GenerateMosaicParams params,
    int k,
    Fl_Progress* progressBar
) {
    std::vector<std::string> filenames;
    cv::Mat featureMatrix = buildFeatureMatrix(datasetStats, params, filenames);

    std::atomic<bool> running = true;

    std::thread kmeansProgress([&running]() {
        const char animation[] = "|/-\\";
        int i = 0;
        while (running) {
            std::cout << "\rKMeans clustering in progress... " << animation[i++ % 4] << std::flush;
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
    });

    cv::Mat clusterLabels;
    cv::Mat centers;
    cv::kmeans(
        featureMatrix,
        k,
        clusterLabels,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 1.0),
        3,
        cv::KMEANS_PP_CENTERS,
        centers
    );

    running = false;
    kmeansProgress.join();
    std::cout << "\rKMeans clustering completed.                           \n";

    std::map<int, std::vector<std::string>> clusterToImages;
    for (int i = 0; i < clusterLabels.rows; ++i) {
        int clusterId = clusterLabels.at<int>(i);
        clusterToImages[clusterId].push_back(filenames[i]);
    }

    std::set<std::string> usedImages;
    cv::Mat mosaic = inputImage.clone();
    int rowBlocks = inputImage.rows / blockSize;
    int colBlocks = inputImage.cols / blockSize;

    for (int i = 0; i < rowBlocks; i++) {
        for (int j = 0; j < colBlocks; j++) {
            cv::Rect roi(j * blockSize, i * blockSize, blockSize, blockSize);
            cv::Mat block = inputImage(roi).clone();

            StatisticalFeatures blockStats = processImageStats(block);
            std::vector<float> blockVec = featuresFromStat(blockStats, params);

            cv::Mat blockMat(1, blockVec.size(), CV_32F);
            for (int z = 0; z < blockVec.size(); ++z)
                blockMat.at<float>(0, z) = blockVec[z];

            int bestCluster = 0;
            float minDist = std::numeric_limits<float>::max();
            for (int c = 0; c < centers.rows; ++c) {
                float dist = cv::norm(blockMat, centers.row(c), cv::NORM_L2);
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = c;
                }
            }

            const auto& clusterImgs = clusterToImages[bestCluster];
            std::string bestImage = "";

            if (params.reuseImages) {
                bestImage = clusterImgs[rand() % clusterImgs.size()];
            } else {
                for (const auto& imgPath : clusterImgs) {
                    if (usedImages.find(imgPath) == usedImages.end()) {
                        bestImage = imgPath;
                        usedImages.insert(imgPath);
                        break;
                    }
                }

                if (bestImage.empty()) {
                    bestImage = clusterImgs[rand() % clusterImgs.size()];
                }
            }

            cv::Mat bestImg = cv::imread(bestImage);
            if (bestImg.empty()) continue;

            cv::resize(bestImg, bestImg, cv::Size(blockSize, blockSize));
            bestImg.copyTo(mosaic(roi));
        }

        std::cout << "Progress : " << int((i * colBlocks) / float(rowBlocks * colBlocks) * 100) << "%" << std::flush << "\r";
        progressBar->value((i * colBlocks) / float(rowBlocks * colBlocks));
    }

    std::cout << "Progress : 100%" << std::endl;
    return mosaic;
}
