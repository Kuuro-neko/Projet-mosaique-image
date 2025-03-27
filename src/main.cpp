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

#define MEANCOLORS_FILE "mean_values.txt"

struct Color{
    double r;
    double g;
    double b;
};

// précalcul des moyennes des couleurs des images du dataset
std::map<std::string, Color> preprocessDatasetMeanColor(const std::string& folderPath){
    std::map<std::string, Color> meanValues;

    for(const auto& entry : fs::directory_iterator(folderPath)){

        cv::Mat img = cv::imread(entry.path().string());

        cv::Scalar mean = cv::mean(img);
        meanValues[entry.path().string()] = {mean[2], mean[1], mean[0]};
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

cv::Mat generateMosaic(const cv::Mat& inputImage, std::map<std::string, Color> &meanValues, int blockSize, bool reuseImages = false){
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
                Color second = entry.second;
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

std::map<std::string, Color> checkIfAlreadyPreProcessed(const std::string& folderPath){
    std::map<std::string, Color> meanValues;

    if(fs::exists(MEANCOLORS_FILE)){
        std::cout << "Loading mean values from file" << std::endl;
        std::ifstream file(MEANCOLORS_FILE);
        std::string line;
        while(std::getline(file, line)){
            std::istringstream iss(line);
            std::string key;
            Color value;
            iss >> key >> value.r >> value.g >> value.b;
            meanValues[key] = value;
        }
        file.close();
    } else {
        std::cout << "Preprocessing the dataset" << std::endl;
        meanValues = preprocessDatasetMeanColor(folderPath);
    
        // Write the mean values to a file
        std::ofstream file(MEANCOLORS_FILE);
        for(const auto& entry : meanValues){
            file << entry.first << " ";
            file << entry.second.r << " ";
            file << entry.second.g << " ";
            file << entry.second.b << std::endl;
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

    std::map<std::string, Color> meanValues = checkIfAlreadyPreProcessed(argv[2]);

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