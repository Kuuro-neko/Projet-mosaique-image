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

/**
 * @brief Parameters for the generation of the mosaic
 * 
 */
struct GenerateMosaicParams{
    bool meanColor = true;
    bool variance = false;
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
    GenerateMosaicParams(bool meanColor = true, bool variance = false, bool skewness = false, bool energy = false, bool reuseImages = false) : meanColor(meanColor), variance(variance), skewness(skewness), energy(energy), reuseImages(reuseImages) {}
    
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

std::string generateResizedDataset(const std::string& folderPath, int blocSize){
    std::cout << "Resizing the dataset" << std::endl;
    int progress = 0;
    int total = std::distance(fs::directory_iterator(folderPath), fs::directory_iterator{});
    std::string resizedFolder = folderPath;
    if (resizedFolder.back() == '/') resizedFolder.pop_back(); // remove the last / if it exists
    resizedFolder += "_resized_" + std::to_string(blocSize);
    if (!fs::exists(resizedFolder)){
        fs::create_directory(resizedFolder);
    }
    for (const auto& entry : fs::directory_iterator(folderPath)){
        try {
            // checkif the resized image already exists
            if (fs::exists(resizedFolder + "/" + entry.path().filename().string())){
                progress++;
                continue;
            }
            cv::Mat img = cv::imread(entry.path().string());
            cv::resize(img, img, cv::Size(blocSize, blocSize));
            cv::imwrite(resizedFolder + "/" + entry.path().filename().string(), img);
            progress++;
            if (progress % 250 == 0){
                std::cout << "Progress : " << int(progress / (float)total * 100) << "%" << std::flush << "\r";
            }
        } catch (cv::Exception& e){
            // There is probably a json file in the dataset folder, catching it's error here
            std::cout << "Error while processing image : " << entry.path().string() << std::endl;
        }
    }
    std::cout << "Progress : 100%" << std::endl;
    return resizedFolder;
}

/**
 * @brief Prétraitement du dataset pour obtenir les statistiques des images
 * 
 * @param folderPath 
 * @return std::map<std::string, StatisticalFeatures> 
 */
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

std::map<std::string, Tamura> preprocessDatasetTamura(const std::string& folderPath){
    std::map<std::string, Tamura> tamuraValues;

    for(const auto& entry : fs::directory_iterator(folderPath)){

        cv::Mat img = cv::imread(entry.path().string());

        // TO DO ?

        tamuraValues[entry.path().string()] = {0, 0, 0}; // TO REPLACE
    }

    return tamuraValues;
}


/**
 * @brief Découpe une image en blocs de taille blockSize
 * 
 * @param image 
 * @param blockSize 
 * @return std::vector<cv::Mat> 
 */
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

/**
 * @brief Calcul de la distance entre deux images
 * 
 * @param a 
 * @param b 
 * @param params 
 * @return double 
 */
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

double computeManhattanDistance(StatisticalFeatures a, StatisticalFeatures b, GenerateMosaicParams params){
    double distance = 0;
    if(params.meanColor){
        distance += abs(a.mean.r - b.mean.r) + abs(a.mean.g - b.mean.g) + abs(a.mean.b - b.mean.b);
    }
    if(params.variance){
        distance += abs(a.variance.r - b.variance.r) + abs(a.variance.g - b.variance.g) + abs(a.variance.b - b.variance.b);
    }
    if(params.skewness){
        distance += abs(a.skewness.r - b.skewness.r) + abs(a.skewness.g - b.skewness.g) + abs(a.skewness.b - b.skewness.b);
    }
    if(params.energy){
        distance += abs(a.energy.r - b.energy.r) + abs(a.energy.g - b.energy.g) + abs(a.energy.b - b.energy.b);
    }
    return distance;
}

/**
 * @brief Génère une mosaïque à partir d'une image et d'un ensemble d'images de référence
 * 
 * @param inputImage 
 * @param meanValues 
 * @param blockSize 
 * @param params 
 * @return cv::Mat 
 */
cv::Mat generateMosaic(const cv::Mat& inputImage, std::map<std::string, StatisticalFeatures> &meanValues, int blockSize, GenerateMosaicParams params){
    std::cout << "Generating mosaic with parameters : block size : " << blockSize << ", " << params.toString() << std::endl;
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

            StatisticalFeatures blockStats = processImageStats(block);

            double minDistance = std::numeric_limits<double>::max();
            std::string bestMatch;

            for(const auto& entry : meanValues){
                StatisticalFeatures second = entry.second;
                double distance = computeDistance(blockStats, second, params);

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
            if (!params.reuseImages) meanValues.erase(bestMatch);
        }

        std::cout << "Progress : " << int((i * colBlocks) / (float)totalBlocks * 100) << "%" << std::flush << "\r";
    }
    std::cout << "Progress : 100%" << std::endl;
    return mosaic;
}

double computeAlignmentScore(const cv::Mat& block, const cv::Mat& reference){

    // compute the alignment score as if the images were a big string of pixel values
    int maxShift = std::max(block.rows, block.cols);
    double maxScore = 0;
    for (int shiftX = -maxShift; shiftX < maxShift; shiftX++)
    {
        for (int shiftY = -maxShift; shiftY < maxShift; shiftY++)
        {
            double score = 0;
            for (int i = 0; i < block.rows; i++)
            {
                for (int j = 0; j < block.cols; j++)
                {
                    int x = j + shiftX;
                    int y = i + shiftY;
                    if (x >= 0 && x < reference.cols && y >= 0 && y < reference.rows){
                        cv::Vec3b blockPixel = block.at<cv::Vec3b>(i, j);
                        cv::Vec3b refPixel = reference.at<cv::Vec3b>(y, x);
                        score += sqrt(pow(blockPixel[0] - refPixel[0], 2) + pow(blockPixel[1] - refPixel[1], 2) + pow(blockPixel[2] - refPixel[2], 2));
                    }
                }
            }
            if (score > maxScore){
                maxScore = score;
            }
        }
    }

    return maxScore;
}

cv::Mat generateMosaicUsingAlignment(const cv::Mat& inputImage, int blockSize, const std::string& folderPath, GenerateMosaicParams params){
    std::cout << "Generating mosaic with parameters : block size : " << blockSize << ", " << params.toString() << std::endl;
    cv::Mat mosaic = inputImage.clone();
    std::vector<cv::Mat> blocks = splitImageIntoBlocks(inputImage, blockSize);

    int rowBlocks = inputImage.rows / blockSize;
    int colBlocks = inputImage.cols / blockSize;

    int totalBlocks = rowBlocks * colBlocks;

    std::string resizedFolder = generateResizedDataset(folderPath, blockSize);
    std::cout << "Resized folder : " << resizedFolder << std::endl;
    for (int i = 0; i < rowBlocks; i++)
    {
        for (int j = 0; j < colBlocks; j++)
        {
            
            cv::Rect roi(j * blockSize, i * blockSize, blockSize, blockSize);
            cv::Mat block = mosaic(roi).clone();

            double maxScore = 0;
            std::string bestMatch;

            for(const auto& entry : fs::directory_iterator(resizedFolder)){
                cv::Mat reference = cv::imread(entry.path().string());
                double score = computeAlignmentScore(block, reference);

                if(score > maxScore){
                    maxScore = score;
                    bestMatch = entry.path().string();
                }
            }

            cv::Mat bestMatchImg = cv::imread(bestMatch);

            // resize de l'imagette pour qu'elle corresponde à la taille du bloc
            cv::resize(bestMatchImg, bestMatchImg, cv::Size(blockSize, blockSize));

            bestMatchImg.copyTo(mosaic(roi));
        }

        std::cout << "Progress : " << int((i * colBlocks) / (float)totalBlocks * 100) << "%" << std::flush << "\r";
    }
    std::cout << "Progress : 100%" << std::endl;
    return mosaic;
}
          

/**
 * @brief Précalcul des statistiques des images du dataset si elles n'ont pas déjà été calculées
 * 
 * @param folderPath 
 * @return std::map<std::string, StatisticalFeatures> 
 */
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
        meanValues = preprocessDatasetStats(folderPath);
    
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

/**
 * @brief Calcul du PSNR entre deux images
 * 
 * @param I1 
 * @param I2 
 * @return float 
 */
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
    if ( argc < 4 || argc > 5 )
    {
        printf("usage: %s <Image_Path> <DATASET_Folder_Path> <Bloc size> [parameters as bit array]\n", argv[0]);
        std::cout << "Parameters : meanColor, variance, skewness, energy, reuseImages. Ex : 10101" << std::endl;
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

    GenerateMosaicParams params;
    if (argc == 5){
        params.setFromBitArray(argv[4]);
    }/* else { // Example, this specific example is unnecessary because of the default constructor
        params.meanColor = true;
        params.variance = false;
        params.skewness = false;
        params.energy = false;
        params.reuseImages = false;
    }*/

    //cv::Mat mosaic = generateMosaic(inputImage, meanValues, blockSize, params);
    cv::Mat mosaic = generateMosaicUsingAlignment(inputImage, blockSize, argv[2], params);

    float psnr = PSNR(inputImage, mosaic);
    std::cout << "Mosaic generated. PSNR : " << psnr << std::endl;

    cv::namedWindow("Mosaïque", cv::WINDOW_AUTOSIZE);
    cv::imshow("Mosaïque", mosaic);
    cv::waitKey(0);

    cv::imwrite("mosaic_output.jpg", mosaic);
    return 0;
}