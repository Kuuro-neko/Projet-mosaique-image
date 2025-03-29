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

struct Color{
    double r;
    double g;
    double b;
};

// chargement base d'imagettes
std::vector<cv::Mat> loadImagesFromFolder(const std::string& folderPath){

    std::vector<cv::Mat> images;

    for(const auto& entry : fs::directory_iterator(folderPath)){

        cv::Mat img = cv::imread(entry.path().string());

        images.push_back(img);
    }

    return images;
}

std::map<std::string, Color> preprocessDataset(const std::string& folderPath){
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
            cv::Rect roi(x, y, blockSize, blockSize);
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

    std::vector<std::vector<double>> distances;
    std::vector<std::vector<std::string>> bestMatches;

    for (int i = 0; i < rowBlocks; i++)
    {
        for (int j = 0; j < colBlocks; j++)
        {
            cv::Rect roi(j * blockSize, i * blockSize, blockSize, blockSize);
            cv::Mat block = mosaic(roi).clone();
            Color mean = {cv::mean(block)[2], cv::mean(block)[1], cv::mean(block)[0]};
            std::vector<std::string> listeMatch;
            std::vector<double> listeDistance;
            for(const auto& entry : meanValues){
                Color second = entry.second;
                double distance = computeDistance(mean, second);
                if(listeDistance.size()==0){
                    listeMatch.push_back(entry.first);
                    listeDistance.push_back(distance);
                }
                else{
                    if(listeDistance[0]>distance){
                        listeDistance.insert(listeDistance.begin(),distance);
                        listeMatch.insert(listeMatch.begin(),entry.first);
                        if(listeDistance.size()>rowBlocks*colBlocks){
                            listeDistance.pop_back();
                            listeMatch.pop_back();
                        }
                    }
                    else{
                        if(distance<listeDistance[listeDistance.size()-1]){
                            bool a=true;
                            for(int m=listeDistance.size()-1;m>=0;m--){
                                if(distance>listeDistance[m]){
                                    listeDistance.insert(listeDistance.begin()+m,distance);
                                    listeMatch.insert(listeMatch.begin()+m,entry.first);
                                    if(listeDistance.size()>rowBlocks*colBlocks){
                                        listeDistance.pop_back();
                                        listeMatch.pop_back();
                                    }
                                    a=false;
                                    break;
                                }
                            }
                            if(listeDistance.size()<rowBlocks*colBlocks){
                                if(a){
                                    listeDistance.push_back(distance);
                                    listeMatch.push_back(entry.first);
                                }
                            }
                        }else{
                            if(listeDistance.size()<rowBlocks*colBlocks){
                                listeDistance.push_back(distance);
                                listeMatch.push_back(entry.first);
                            }
                        }
                    }
                }
            }
            distances.push_back(listeDistance);
            bestMatches.push_back(listeMatch);
            // std::cout << "longueur distance " << listeDistance.size() << std::endl;
            // std::cout << "distance min : " << listeDistance[0] << "distance max : " << listeDistance[listeDistance.size()-1] << std::endl;
        }
    }

    std::vector<std::string> listeMatch;
    std::vector<std::vector<std::vector<double>>> listeValMatch;

    for(int i=0;i<rowBlocks;i++){
        for(int j=0;j<colBlocks;j++){
            int taille=i*rowBlocks+j;
            for(int k=0;k<distances[taille].size();k++){
                std::vector<double> a{distances[taille][k],i,j};
                auto b=std::find(listeMatch.begin(),listeMatch.end(),bestMatches[taille][k]);
                if(b==listeMatch.end()){
                    listeMatch.push_back(bestMatches[taille][k]);
                    std::vector<std::vector<double>> c;
                    c.push_back(a);
                    listeValMatch.push_back(c);
                }else{
                    int c=std::distance(listeMatch.begin(),b);
                    listeValMatch[c].push_back(a);
                }
            }
        }
    }
    std::cout<<listeMatch.size()<<std::endl;


    std::vector<std::string> imageUsed;
    std::vector<std::vector<std::string>> imageFinale;
    std::vector<std::vector<bool>> blockBool;
    for (int i = 0; i < rowBlocks; i++)
    {
        blockBool.push_back(std::vector<bool>());
        imageFinale.push_back(std::vector<std::string>());
        for (int j = 0; j < colBlocks; j++)
        {
            blockBool[i].push_back(false);
            imageFinale[i].push_back("");
        }
    }
    while(imageUsed.size()<distances.size()){
        for(int i=0;i<rowBlocks;i++){
            for(int j=0;j<colBlocks;j++){
                // std::cout<<i<<" "<<j<<std::endl;
                if(blockBool[i][j]==false){
                    int k=0;
                    // std::cout<<"i"<<std::endl;
                    while(std::find(imageUsed.begin(),imageUsed.end(),bestMatches[i*rowBlocks+j][k])!=imageUsed.end()){
                        k++;
                    }
                    std::string bestMatch=bestMatches[i*rowBlocks+j][k];
                    auto a=std::find(listeMatch.begin(),listeMatch.end(),bestMatch);
                    int l=std::distance(listeMatch.begin(),a);
                    bool boo=true;
                    // std::cout<<"k "<<l<<std::endl;
                    for(int m=0;m<listeValMatch[l].size();m++){
                        // std::cout<<"k "<<listeValMatch.size()<<std::endl;
                        // std::cout<<"k "<<listeValMatch[k][m][2]<<std::endl;
                        if(blockBool[listeValMatch[l][m][1]][listeValMatch[l][m][2]]==false && listeValMatch[l][m][1]*rowBlocks+listeValMatch[l][m][2]!=i*rowBlocks+j){
                            boo=false;
                            break;
                        }
                        if(listeValMatch[l][m][1]*rowBlocks+listeValMatch[l][m][2]==i*rowBlocks+j){
                            break;
                        }
                    }
                    // std::cout<<"l "<<std::endl;
                    if(boo){
                        blockBool[i][j]=true;
                        imageUsed.push_back(bestMatch);
                        imageFinale[i][j]=bestMatch;
                    }
                }
            }
        }
    }

    for (int i = 0; i < rowBlocks; i++)
    {
        for (int j = 0; j < colBlocks; j++)
        {
            // std::cout << "Computing block " << i << " " << j << std::endl;

            cv::Rect roi(j * blockSize, i * blockSize, blockSize, blockSize);
            cv::Mat block = mosaic(roi).clone();

            Color mean = {cv::mean(block)[2], cv::mean(block)[1], cv::mean(block)[0]};

            std::string bestMatch;
            
            // for(int k=0;k<distances.size();k++){
            //     if(std::find(imageUsed.begin(),imageUsed.end(),bestMatches[i*rowBlocks+j][k])==imageUsed.end()){
            //         double minDistance = std::numeric_limits<double>::max();
            //         auto a=std::find(listeMatch.begin(),listeMatch.end(),bestMatches[i*rowBlocks+j][k]);
            //         int l=std::distance(listeMatch.begin(),a);
            //         bestMatch=bestMatches[i*rowBlocks+j][k];
            //         // std::cout<<listeValMatch[l].size()<<std::endl;
            //         for(int m=0;m<listeValMatch[l].size();m++){
            //             if(listeValMatch[l][m][0]<minDistance){
            //                 minDistance=listeValMatch[l][m][0];
            //                 // bestMatch=bestMatches[listeValMatch[l][m][1]][listeValMatch[l][m][2]];
            //             }
            //         }
            //         if(distances[i*rowBlocks+j][k]<=minDistance){
            //             break;
            //         }
            //     }else{
            //         // std::cout << "image déjà utilisée" << std::endl;
            //     }
            // }
            
            // imageUsed.push_back(bestMatch);

            // std::cout<<"longueur imageUsed : "<<imageUsed.size()<<std::endl;

            // cv::Mat bestMatchImg = cv::imread(bestMatch);
            cv::Mat bestMatchImg = cv::imread(imageFinale[i][j]);

            // resize de l'imagette pour qu'elle corresponde à la taille du bloc
            cv::resize(bestMatchImg, bestMatchImg, cv::Size(blockSize, blockSize));


            bestMatchImg.copyTo(mosaic(roi));

            // remove used image from the map
            if (!reuseImages) meanValues.erase(bestMatch);
        }
    }

    std::cout<<"listeMatch : "<<listeMatch.size()<<std::endl;

    // for (int i = 0; i < rowBlocks; i++)
    // {
    //     for (int j = 0; j < colBlocks; j++)
    //     {
    //         std::cout << "Computing block " << i << " " << j << std::endl;

    //         cv::Rect roi(j * blockSize, i * blockSize, blockSize, blockSize);
    //         cv::Mat block = mosaic(roi).clone();

    //         Color mean = {cv::mean(block)[2], cv::mean(block)[1], cv::mean(block)[0]};

    //         double minDistance = std::numeric_limits<double>::max();
    //         std::string bestMatch;

    //         for(const auto& entry : meanValues){
    //             Color second = entry.second;
    //             double distance = computeDistance(mean, second);

    //             if(distance < minDistance){
    //                 minDistance = distance;
    //                 bestMatch = entry.first;
    //             }
    //         }

    //         cv::Mat bestMatchImg = cv::imread(bestMatch);

    //         // resize de l'imagette pour qu'elle corresponde à la taille du bloc
    //         cv::resize(bestMatchImg, bestMatchImg, cv::Size(blockSize, blockSize));


    //         bestMatchImg.copyTo(mosaic(roi));

    //         // remove used image from the map
    //         if (!reuseImages) meanValues.erase(bestMatch);
    //     }
        
    // }

    
    return mosaic;
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
    
    // if mean_values.txt exists, load the mean values from the file
    std::map<std::string, Color> meanValues;
    if(fs::exists("mean_values.txt")){
        std::cout << "Loading mean values from file" << std::endl;
        std::ifstream file("mean_values.txt");
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
        meanValues = preprocessDataset(argv[2]);
    
        // Write the mean values to a file
        std::ofstream file("mean_values.txt");
        for(const auto& entry : meanValues){
            file << entry.first << " ";
            file << entry.second.r << " ";
            file << entry.second.g << " ";
            file << entry.second.b << std::endl;
        }
        file.close();
    }

    std::cout << "Generating mosaic" << std::endl;

    cv::Mat mosaic = generateMosaic(inputImage, meanValues, blockSize);
    
    // cv::namedWindow("Mosaïque", cv::WINDOW_AUTOSIZE);
    std::cout<<"i"<<std::endl;
    // cv::imshow("Mosaïque", mosaic);
    // cv::waitKey(0);

    cv::imwrite("mosaic_output2.jpg", mosaic);


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