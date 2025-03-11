#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem> 

using namespace cv;
namespace fs = std::filesystem;

// chargement base d'imagettes
std::vector<cv::Mat> loadImagesFromFolder(const std::string& folderPath){

    std::vector<cv::Mat> images;

    for(const auto& entry : fs::directory_iterator(folderPath)){

        cv::Mat img = cv::imread(entry.path().string());

        if(img.empty()){
            printf("Impossible de charger ", entry.path());
            continue;
        }

        images.push_back(img);
    }

    return images;
}

//retourner une image carrée
cv::Mat extractCenteredSquare(const cv::Mat& image){
    int size = std::min(image.cols, image.rows); 

    cv::Rect roi(0,0, size, size);
    image = image(roi).clone();

    return image;
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

double computeDistance(const cv::Mat& img1, const cv::Mat& img2){

    cv::Scalar mean1 = cv::mean(img1);
    cv::Scalar mean2 = cv::mean(img2);

    double distance = sqrt(pow(mean1[0] - mean2[0], 2) + pow(mean1[1] - mean2[1], 2) + pow(mean1[2] - mean2[2], 2)); // B -- V -- R

    return distance;
}

//génération de mosaique
cv::Mat generateMosaic(const cv::Mat& inputImage, const std::vector<cv::Mat>& tileImages, int blockSize){

    cv::Mat mosaic = inputImage.clone();
    std::vector<cv::Mat> blocks = splitImageIntoBlocks(inputImage, blockSize);

    for(size_t i = 0; i < blocks.size(); i++){
        double minDistance = std::numeric_limits<double>::max();
        int bestTileIndex = 0;

        //Trouver l'imagette la plus similaire
        for(size_t j = 0; j < tileImages.size(); j++){
            double distance = computeDistance(blocks[i], tileImages[j]);
            if(distance < minDistance){
                minDistance = distance;
                bestTileIndex = j;
            }
        }

        //Remplacer le block par l'imagette séléctionnée
        int x = (i % (inputImage.cols / blockSize)) * blockSize;
        int y = (i / (inputImage.cols / blockSize)) * blockSize;
        tileImages[bestTileIndex].copyTo(mosaic(cv::Rect(x, y, blockSize, blockSize)));
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

    std::vector<cv::Mat> tileImages = loadImagesFromFolder(argv[2]);
    if(tileImages.empty()) {
        printf("Aucune imagette trouvée.");
        return -1;
    }

    //générer la mosaique
    int blockSize = 32;
    cv::Mat mosaic = generateMosaic(inputImage, tileImages, blockSize);

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