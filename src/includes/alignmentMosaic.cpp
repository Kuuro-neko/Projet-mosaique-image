#include "../includes/alignmentMosaic.hpp"
#include <algorithm>

int alignmentScoreEfficient(const std::string &a, const std::string &b, int match, int mismatch, int gap, int n, int **score) {
    // Needleman–Wunsch algorithm
    for (int i = 0; i <= n; i++) {
        score[i][0] = i * gap;
    }
    for (int j = 0; j <= n; j++) {
        score[0][j] = j * gap;
    }
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            int diag = score[i - 1][j - 1] + (a[i - 1] == b[j - 1] ? match : mismatch);
            int up = score[i - 1][j] + gap;
            int left = score[i][j - 1] + gap;
            score[i][j] = std::max({diag, up, left});
        }
    }
    int scoreValue = score[n][n];
    return scoreValue;
}

int alignmentScoreRowByRowEfficient(const std::vector<std::string> &a, const std::vector<std::string> &b, int match, int mismatch, int gap, int n, int **score) {
    int out = 0;
    for (int i = 0; i < a.size(); i++) {
        out += alignmentScoreEfficient(a[i], b[i], match, mismatch, gap, n, score);
    }
    return out;
}

std::string getImageAsString(const cv::Mat& img){
    std::ostringstream oss;
    const uchar* data = img.data;
    for (size_t i = 0; i < img.total() * img.elemSize(); ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(data[i]);
    }
    return oss.str();
}

std::vector<std::string> getImageRowsAsString(const cv::Mat& img){
    std::vector<std::string> rows;
    for (int i = 0; i < img.rows; i++){
        std::ostringstream oss;
        const uchar* data = img.ptr<uchar>(i);
        for (size_t j = 0; j < img.cols * img.elemSize(); ++j) {
            oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(data[j]);
        }
        rows.push_back(oss.str());
    }
    return rows;
}

std::string generatePrecomputedStringsRows(const std::string& folderPath, int blocksize, int n){
    std::cout << " Generating precomputed strings for the resized images" << std::endl;
    std::cout << "  Folder path : " << folderPath << std::endl;
    std::cout << "  Number of images to be kept : " << n << std::endl;
    int progress = 0;
    // For each image in the folder, resize it to the block size and save it in a new folder like this :
    // [1st row hex string] [2nd row hex string] [3rd row hex string] ... [blocksize-th row hex string]
    std::string precomputedStringsFile = "_precomputed_strings_" + std::to_string(blocksize) + ".txt";
    if (fs::exists(precomputedStringsFile)){
        std::cout << "  Precomputed strings file already exists, skipping generation" << std::endl;
        return precomputedStringsFile;
    }
    std::ofstream file(precomputedStringsFile);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << precomputedStringsFile << std::endl;
        return precomputedStringsFile;
    }
    int total = n == -1 ? std::distance(fs::directory_iterator(folderPath), fs::directory_iterator{}) : n;
    for (const auto& entry : fs::directory_iterator(folderPath)){
        try {
            cv::Mat img = cv::imread(entry.path().string());
            if (img.empty()){
                std::cout << "Error while reading image : " << entry.path().string() << std::endl;
                continue;
            }
            cv::resize(img, img, cv::Size(blocksize, blocksize));
            std::vector<std::string> rows = getImageRowsAsString(img);
            for (const auto& row : rows){
                file << row << " ";
            }
            file << std::endl;
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
    return precomputedStringsFile;
}

std::vector<std::vector<std::string>> loadPrecomputedStringsRows(const std::string& precomputedStringsFile){
    std::vector<std::vector<std::string>> precomputedStringsRows;
    std::ifstream file(precomputedStringsFile);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << precomputedStringsFile << std::endl;
        return precomputedStringsRows;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<std::string> row;
        std::string value;
        while (iss >> value) {
            row.push_back(value);
        }
        precomputedStringsRows.push_back(row);
    }
    file.close();
    return precomputedStringsRows;
}

cv::Mat loadImageFromHexString(const std::string& hexString, int rows, int cols){
    cv::Mat img(rows, cols, CV_8UC3);
    for (size_t i = 0; i < hexString.length(); i += 6) {
        std::string hexValue = hexString.substr(i, 6);
        int r = std::stoi(hexValue.substr(0, 2), nullptr, 16);
        int g = std::stoi(hexValue.substr(2, 2), nullptr, 16);
        int b = std::stoi(hexValue.substr(4, 2), nullptr, 16);
        img.at<cv::Vec3b>(i / 6 / cols, (i / 6) % cols) = cv::Vec3b(r, g, b);
    }
    return img;
}

std::string assembleStringsRows(const std::vector<std::string>& strings){
    std::ostringstream oss;
    for (const auto& str : strings){
        oss << str;
    }
    return oss.str();
}

cv::Mat generateMosaicUsingAlignment(const cv::Mat& inputImage, int blockSize, const std::string& folderPath, bool uniquesImagettes){
    std::cout << "==== Generating mosaic using alignment method, block size : " << blockSize << ", uniquesImagettes : " << uniquesImagettes << " ====" << std::endl;
    cv::Mat mosaic = inputImage.clone();

    int rowBlocks = inputImage.rows / blockSize;
    int colBlocks = inputImage.cols / blockSize;

    int totalBlocks = rowBlocks * colBlocks;

    std::string precomputedStringsFile = generatePrecomputedStringsRows(folderPath, blockSize, 1000);
    std::cout << "Precomputed strings file : " << precomputedStringsFile << std::endl;

    std::vector<std::vector<std::string>> precomputedStringsRows = loadPrecomputedStringsRows(precomputedStringsFile);
    std::vector<std::string> inputBlocStringsRows;

    int **scoreTab = new int *[blockSize*6 + 1];
    for (int i = 0; i <= blockSize*6; i++) {
        scoreTab[i] = new int[blockSize*6 + 1];
        scoreTab[i][0] = i * -1;
    }
    std::cout << "Progress : " << int(0 / (float)totalBlocks * 100) << "%" << std::flush << "\r";
    for (int i = 0; i < rowBlocks; i++)
    {
        for (int j = 0; j < colBlocks; j++)
        {
            cv::Rect roi(j * blockSize, i * blockSize, blockSize, blockSize);
            cv::Mat block = mosaic(roi).clone();
            inputBlocStringsRows = getImageRowsAsString(block);
            int rowLength = inputBlocStringsRows[0].length();

            int maxScore = std::numeric_limits<int>::min();
            std::vector<std::string>* bestMatch;

            for(auto& precomputedStringsRow : precomputedStringsRows) {
                int score = alignmentScoreRowByRowEfficient(precomputedStringsRow, inputBlocStringsRows, 1, -1, -1, rowLength, scoreTab);
                if (score > maxScore) {
                    maxScore = score;
                    bestMatch = &precomputedStringsRow;
                }
            }
            cv::Mat bestMatchImg = loadImageFromHexString(assembleStringsRows(*bestMatch), blockSize, blockSize);
            bestMatchImg.copyTo(mosaic(roi));
        }

        std::cout << "Progress : " << int((i * colBlocks) / (float)totalBlocks * 100) << "%" << std::flush << "\r";
    }
    std::cout << "Progress : 100%" << std::endl;

    for (int i = 0; i <= blockSize*6; i++) {
        delete[] scoreTab[i];
    }
    delete[] scoreTab;

    return mosaic;
}