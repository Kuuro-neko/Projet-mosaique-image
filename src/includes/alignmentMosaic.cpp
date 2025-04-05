#include "../includes/alignmentMosaic.hpp"
#include <algorithm>

int alignmentScore(const std::string &a, const std::string &b, int match, int mismatch, int gap) {
    int n = a.size();
    int m = b.size();

    // Needleman–Wunsch algorithm
    int **score = new int *[n + 1];

    for (int i = 0; i <= n; i++) {
        score[i] = new int[m + 1];
        score[i][0] = i * gap;
    }
    for (int j = 0; j <= m; j++) {
        score[0][j] = j * gap;
    }
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            int diag = score[i - 1][j - 1] + (a[i - 1] == b[j - 1] ? match : mismatch);
            int up = score[i - 1][j] + gap;
            int left = score[i][j - 1] + gap;
            score[i][j] = std::max({diag, up, left});
        }
    }

    int scoreValue = score[n][m];

    // Free the memory
    for (int i = 0; i <= n; i++) {
        delete[] score[i];
    }
    delete[] score;

    return scoreValue;
}

std::string getImageAsString(const cv::Mat& img){
    // convert the image to a string
    std::ostringstream oss;
    const uchar* data = img.data;
    for (size_t i = 0; i < img.total() * img.elemSize(); ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(data[i]);
    }
    return oss.str();
}

std::string generateResizedDataset(const std::string& folderPath, int blocSize, int n){
    std::cout << "Resizing the images from the dataset" << std::endl;
    std::cout << "Bloc size : " << blocSize << std::endl;
    std::cout << "Number of images to be kept : " << n << std::endl;
    int progress = 0;
    int total = n == -1 ? std::distance(fs::directory_iterator(folderPath), fs::directory_iterator{}) : n;
    std::string resizedFolder = folderPath;
    if (resizedFolder.back() == '/') resizedFolder.pop_back(); // remove the last / if it exists
    resizedFolder += "_resized_" + std::to_string(blocSize) + "_" + std::to_string(n);
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
            if (n != -1 && progress >= n){
                break;
            }
        } catch (cv::Exception& e){
            // There is probably a json file in the dataset folder, catching it's error here
            std::cout << "Error while processing image : " << entry.path().string() << std::endl;
        }
    }
    std::cout << "Progress : 100%" << std::endl;
    return resizedFolder;
}

cv::Mat generateMosaicUsingAlignment(const cv::Mat& inputImage, int blockSize, const std::string& folderPath, bool uniquesImagettes){
    std::cout << "Generating mosaic using alignment method, block size : " << blockSize << ", uniquesImagettes : " << uniquesImagettes << std::endl;
    cv::Mat mosaic = inputImage.clone();

    int rowBlocks = inputImage.rows / blockSize;
    int colBlocks = inputImage.cols / blockSize;

    int totalBlocks = rowBlocks * colBlocks;

    std::string resizedFolder = generateResizedDataset(folderPath, blockSize, 500);
    std::cout << "Resized folder : " << resizedFolder << std::endl;
    for (int i = 0; i < rowBlocks; i++)
    {
        for (int j = 0; j < colBlocks; j++)
        {
            
            cv::Rect roi(j * blockSize, i * blockSize, blockSize, blockSize);
            cv::Mat block = mosaic(roi).clone();
            std::string blockAsString = getImageAsString(block);

            int maxScore = -INFINITY;
            std::string bestMatch;

            for(const auto& entry : fs::directory_iterator(resizedFolder)){
                cv::Mat reference = cv::imread(entry.path().string());
                int score = alignmentScore(blockAsString, getImageAsString(reference));

                if(score > maxScore){
                    maxScore = score;
                    bestMatch = entry.path().string();
                }
            }
            cv::Mat bestMatchImg = cv::imread(bestMatch);
            bestMatchImg.copyTo(mosaic(roi));
        }

        std::cout << "Progress : " << int((i * colBlocks) / (float)totalBlocks * 100) << "%" << std::flush << "\r";
    }
    std::cout << "Progress : 100%" << std::endl;
    return mosaic;
}