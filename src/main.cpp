#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem> 
#include <map>
#include <fstream>
#include <sstream>

using namespace cv;
namespace fs = std::filesystem;

#define STATISTICAL_FEATURES_FILE "stats_features.txt"

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

struct Tamura{
    double coarseness;
    double contrast;
    double directionality;
};

// précalcul des moyennes des couleurs des images du dataset
std::map<std::string, StatisticalFeatures> preprocessDatasetMeanColor(const std::string& folderPath){
    std::map<std::string, StatisticalFeatures> meanValues;

    int progress = 0;
    int total = std::distance(fs::directory_iterator(folderPath), fs::directory_iterator{});

    for(const auto& entry : fs::directory_iterator(folderPath)){
        try {
            cv::Mat img = cv::imread(entry.path().string());
    
            // Color
            cv::Scalar mean = cv::mean(img);
            Color c = {mean[2], mean[1], mean[0]};
    
            // Variance
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
            
            meanValues[entry.path().string()] = {c, v, s, e};
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

std::map<std::string, Tamura> preprocessDatasetTamura(const std::string& folderPath){
    std::map<std::string, Tamura> tamuraValues;

    for(const auto& entry : fs::directory_iterator(folderPath)){

        cv::Mat img = cv::imread(entry.path().string());



        tamuraValues[entry.path().string()] = {0, 0, 0};
    }

    return tamuraValues;
}


// découpage en bloc
std::vector<cv::Mat> splitImageIntoBlocks(const cv::Mat& image, int blockSize){
    std::vector<cv::Mat> blocks;

    int height = image.rows;
    int width = image.cols;

    for (int y = 0; y < height; y += blockSize) {
        for (int x = 0; x < width; x+= blockSize)
        {
            int blockWidth = std::min(blockSize, width - x);
            int blockHeight = std::min(blockSize, height - y);

            cv::Rect roi(x, y, blockWidth, blockHeight);
            cv::Mat block = image(roi).clone();
            blocks.push_back(block);
        } 
    }

    return blocks;
    
}

double computeDistance(Color a, Color b){
    return sqrt(pow(a.r - b.r, 2) + pow(a.g - b.g, 2) + pow(a.b - b.b, 2));
}

cv::Mat generateMosaic(const cv::Mat& inputImage, std::map<std::string, StatisticalFeatures> &meanValues, int blockSize, bool reuseImages = false){
    cv::Mat mosaic = inputImage.clone();
    std::vector<cv::Mat> blocks = splitImageIntoBlocks(inputImage, blockSize);

    int rowBlocks = inputImage.rows / blockSize;
    int colBlocks = inputImage.cols / blockSize;

    int totalBlocks = rowBlocks * colBlocks;

    for (int i = 0; i < rowBlocks; i++)
    {
        for (int j = 0; j < colBlocks; j++)
        {

            cv::Rect roi(j * blockSize, i * blockSize, blockSize, blockSize);
            cv::Mat block = mosaic(roi).clone();

            Color mean = {cv::mean(block)[2], cv::mean(block)[1], cv::mean(block)[0]};

            double minDistance = std::numeric_limits<double>::max();
            std::string bestMatch;

            for(const auto& entry : meanValues){
                Color second = entry.second.mean;
                double distance = computeDistance(mean, second);

                if(distance < minDistance){
                    minDistance = distance;
                    bestMatch = entry.first;
                }
            }

            cv::Mat bestMatchImg = cv::imread(bestMatch);

            // resize de l'imagette pour qu'elle corresponde à la taille du bloc
            cv::resize(bestMatchImg, bestMatchImg, cv::Size(blockSize, blockSize));


            bestMatchImg.copyTo(mosaic(roi));

            // remove used image from the map
            if (!reuseImages) meanValues.erase(bestMatch);
        }

        std::cout << "Progress : " << int((i * colBlocks) / (float)totalBlocks * 100) << "%" << std::flush << "\r";
    }
    std::cout << "Progress : 100%" << std::endl;
    return mosaic;
}

std::map<std::string, StatisticalFeatures> checkIfAlreadyPreProcessed(const std::string& folderPath){
    std::map<std::string, StatisticalFeatures> meanValues;

    if(fs::exists(STATISTICAL_FEATURES_FILE)){
        std::cout << "Loading statistical features from file" << std::endl;
        std::ifstream file(STATISTICAL_FEATURES_FILE);
        std::string line;
        while(std::getline(file, line)){
            std::istringstream iss(line);
            std::string key;
            Color c;
            Variance v;
            Skewness s;
            Energy e;
            iss >> key >> c.r >> c.g >> c.b >> v.r >> v.g >> v.b >> s.r >> s.g >> s.b >> e.r >> e.g >> e.b;
            meanValues[key] = {c, v, s, e};
        }
        file.close();
    } else {
        std::cout << "Preprocessing the dataset" << std::endl;
        meanValues = preprocessDatasetMeanColor(folderPath);
    
        // Write the data to a file
        std::ofstream file(STATISTICAL_FEATURES_FILE);
        for(const auto& entry : meanValues){
            file << entry.first << " " << entry.second.mean.r << " " << entry.second.mean.g << " " << entry.second.mean.b << " "
                 << entry.second.variance.r << " " << entry.second.variance.g << " " << entry.second.variance.b << " "
                 << entry.second.skewness.r << " " << entry.second.skewness.g << " " << entry.second.skewness.b << " "
                 << entry.second.energy.r << " " << entry.second.energy.g << " " << entry.second.energy.b << std::endl;
        }
        file.close();
    }

    return meanValues;
}

float PSNR(const cv::Mat& I1, const cv::Mat& I2)
{
    cv::Mat s1;
    cv::absdiff(I1, I2, s1);   // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    cv::Scalar s = cv::sum(s1);

    double sse = s.val[0] + s.val[1] + s.val[2];

    if( sse <= 1e-10)
        return 0;
    else
    {
        double mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

int main(int argc, char** argv )
{
    if ( argc != 4 )
    {
        printf("usage: %s <Image_Path> <DATASET_Folder_Path> <Bloc size>\n", argv[0]);
        return -1;
    }

    cv::Mat inputImage = cv::imread(argv[1], cv::IMREAD_COLOR);
    if(inputImage.empty()) {
        printf("Impossible de charger l'image originale.");
        return -1;
    }

    int blockSize = std::stoi(argv[3]);

    if (inputImage.size().width % blockSize != 0 || inputImage.size().height % blockSize != 0){
        std::cout << "La taille de l'image n'est pas un multiple de la taille du bloc : " << blockSize << ", taille de l'image : " << inputImage.size() << std::endl;
        return -1;
    }

    std::cout << "Loaded the image : " << argv[1] << " of size : " << inputImage.size() << std::endl;

    std::map<std::string, StatisticalFeatures> meanValues = checkIfAlreadyPreProcessed(argv[2]);

    std::cout << "Generating mosaic" << std::endl;
    cv::Mat mosaic = generateMosaic(inputImage, meanValues, blockSize);
    float psnr = PSNR(inputImage, mosaic);
    std::cout << "Mosaic generated. PSNR : " << psnr << std::endl;

    cv::namedWindow("Mosaïque", cv::WINDOW_AUTOSIZE);
    cv::imshow("Mosaïque", mosaic);
    cv::waitKey(0);

    cv::imwrite("mosaic_output.jpg", mosaic);


    /*Mat image;
    image = imread( argv[1], IMREAD_COLOR );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);

    waitKey(0);*/

    return 0;
}