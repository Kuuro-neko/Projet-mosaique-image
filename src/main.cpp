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

// chargement base d'imagettes
std::vector<cv::Mat> loadImagesFromFolder(const std::string& folderPath){

    std::vector<cv::Mat> images;

    for(const auto& entry : fs::directory_iterator(folderPath)){

        cv::Mat img = cv::imread(entry.path().string());

        images.push_back(img);
    }

    return images;
}

std::map<std::string, double> preprocessDataset(const std::string& folderPath){
    std::map<std::string, double> meanValues;

    for(const auto& entry : fs::directory_iterator(folderPath)){

        cv::Mat img = cv::imread(entry.path().string());

        cv::Scalar mean = cv::mean(img);
        meanValues[entry.path().string()] = mean[0];
    }

    return meanValues;
}


// découpage en bloc
std::vector<cv::Mat> splitImageIntoBlocks(const cv::Mat& image, int blockSize){
    std::vector<cv::Mat> blocks;

    int height = image.rows;
    int width = image.cols;

    for (int y = 0; y < height; y += blockSize) {
        for (int x = 0; x < width; x+= blockSize)
        {
            cv::Rect roi(x, y, blockSize, blockSize);
            cv::Mat block = image(roi).clone();
            blocks.push_back(block);
        } 
    }

    return blocks;
    
}

double computeDistance(double a, double b){
    return std::abs(a - b);
}

cv::Mat generateMosaic(const cv::Mat& inputImage, const std::map<std::string, double> &meanValues, int blockSize){

    cv::Mat mosaic = inputImage.clone();
    std::vector<cv::Mat> blocks = splitImageIntoBlocks(inputImage, blockSize);

    int rowBlocks = inputImage.rows / blockSize;

    int colBlocks = inputImage.cols / blockSize;

    

    for (int i = 0; i < rowBlocks; i++)
    {
        for (int j = 0; j < colBlocks; j++)
        {
            std::cout << "Computing block " << i << " " << j << std::endl;

            cv::Rect roi(j * blockSize, i * blockSize, blockSize, blockSize);
            cv::Mat block = mosaic(roi).clone();

            cv::Scalar mean = cv::mean(block);

            double minDistance = std::numeric_limits<double>::max();
            std::string bestMatch;

            for(const auto& entry : meanValues){
                double second = entry.second;
                double distance = computeDistance(mean[0], second);

                if(distance < minDistance){
                    minDistance = distance;
                    bestMatch = entry.first;
                }
            }

            cv::Mat bestMatchImg = cv::imread(bestMatch);

            // resize de l'imagette pour qu'elle corresponde à la taille du bloc
            cv::resize(bestMatchImg, bestMatchImg, cv::Size(blockSize, blockSize));


            bestMatchImg.copyTo(mosaic(roi));
        }
        
    }

    return mosaic;
}

int main(int argc, char** argv )
{
    if ( argc != 3 )
    {
        printf("usage: %s <Image_Path> <DATASET_Folder_Path>\n", argv[0]);
        return -1;
    }

    cv::Mat inputImage = cv::imread(argv[1], cv::IMREAD_COLOR);
    if(inputImage.empty()) {
        printf("Impossible de charger l'image originale.");
        return -1;
    }

    std::cout << "Loaded the image : " << argv[1] << " of size : " << inputImage.size() << std::endl;
    
    // if mean_values.txt exists, load the mean values from the file
    std::map<std::string, double> meanValues;
    if(fs::exists("mean_values.txt")){
        std::cout << "Loading mean values from file" << std::endl;
        std::ifstream file("mean_values.txt");
        
        std::string line;
        while(std::getline(file, line)){
            std::istringstream iss(line);
            std::string key;
            double value;
            iss >> key >> value;
            meanValues[key] = value;
        }
        file.close();
    } else {
        std::cout << "Preprocessing the dataset" << std::endl;
        meanValues = preprocessDataset(argv[2]);
    
        // Write the mean values to a file
        std::ofstream file("mean_values.txt");
        for(const auto& entry : meanValues){
            file << entry.first << " " << entry.second << std::endl;
        }
        file.close();
    }


    //générer la mosaique
    int blockSize = 32;
    cv::Mat mosaic = generateMosaic(inputImage, meanValues, blockSize);

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