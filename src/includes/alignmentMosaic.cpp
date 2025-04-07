#include "../includes/alignmentMosaic.hpp"
#include <algorithm>

/**
 * @brief Compute the alignment score of two images row using the Needleman-Wunsch algorithm
 * 
 * @param a 
 * @param b 
 * @param n Size of the pixel strings
 * @param score Array of size [n+1][n+1] to store the scores. Memory is managed outside to allow reuse
 * @return int 
 */
int alignmentScore(const std::string &a, const std::string &b, int n, int **score) {
    // Needleman–Wunsch algorithm
    for (int i = 0; i <= n; i++) {
        score[i][0] = i * (-30);
    }
    for (int j = 0; j <= n; j++) {
        score[0][j] = j * (-30);
    }
    int diag, up, left, aR, aG, aB, bR, bG, bB;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            aR = std::stoi(a.substr((i-1)*6, 2), nullptr, 16);
            aG = std::stoi(a.substr((i-1)*6 + 2, 2), nullptr, 16);
            aB = std::stoi(a.substr((i-1)*6 + 4, 2), nullptr, 16);
            bR = std::stoi(b.substr((j-1)*6, 2), nullptr, 16);
            bG = std::stoi(b.substr((j-1)*6 + 2, 2), nullptr, 16);
            bB = std::stoi(b.substr((j-1)*6 + 4, 2), nullptr, 16);
            diag = score[i - 1][j - 1] - ((std::abs(aR - bR) + std::abs(aG - bG) + std::abs(aB - bB)));
            up = score[i - 1][j] - 30;
            left = score[i][j - 1] - 30;
            score[i][j] = std::max({diag, up, left});
        }
    }
    int scoreValue = score[n][n];
    return scoreValue;
}

int alignmentScore(const cv::Mat &a, const cv::Mat &b, int n, int **score) {
    // Needleman–Wunsch algorithm
    for (int i = 0; i <= n; i++) {
        score[i][0] = i * (-30);
    }
    for (int j = 0; j <= n; j++) {
        score[0][j] = j * (-30);
    }
    int diag, up, left, aR, aG, aB, bR, bG, bB;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            aR = a.at<cv::Vec3b>(i - 1)[0];
            aG = a.at<cv::Vec3b>(i - 1)[1];
            aB = a.at<cv::Vec3b>(i - 1)[2];
            bR = b.at<cv::Vec3b>(j - 1)[0];
            bG = b.at<cv::Vec3b>(j - 1)[1];
            bB = b.at<cv::Vec3b>(j - 1)[2];
            diag = score[i - 1][j - 1] - ((std::abs(aR - bR) + std::abs(aG - bG) + std::abs(aB - bB)));
            up = score[i - 1][j] - 30;
            left = score[i][j - 1] - 30;
            score[i][j] = std::max({diag, up, left});
        }
    }
    int scoreValue = score[n][n];
    return scoreValue;
}

/**
 * @brief Compute the alignment score of two images using the Needleman-Wunsch algorithm
 * 
 * @param a 
 * @param b 
 * @param n Size of the pixel strings
 * @param score Array of size [n+1][n+1] to store the scores. Memory is managed outside to allow reuse
 * @return int 
 */
int alignmentScoreRowByRow(const std::vector<std::string> &a, const std::vector<std::string> &b, int n, int **score) {
    int out = 0;
    for (int i = 0; i < a.size(); i++) {
        out += alignmentScore(a[i], b[i], n, score);
    }
    return out;
}

int alignmentScoreRowByRow(const cv::Mat &a, const cv::Mat &b, int n, int **score) {
    int out = 0;
    for (int i = 0; i < a.rows; i++) {
        out += alignmentScore(a.row(i), b.row(i), n, score);
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
            cv::resize(img, img, cv::Size(blocksize, blocksize), cv::INTER_LINEAR);
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
        if (i + 6 > hexString.size()) break;
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

    std::string precomputedStringsFile = generatePrecomputedStringsRows(folderPath, blockSize, 10000);
    std::cout << "Precomputed strings file : " << precomputedStringsFile << std::endl;

    std::vector<std::vector<std::string>> precomputedStringsRows = loadPrecomputedStringsRows(precomputedStringsFile);
    std::vector<cv::Mat> precomputedImages;
    for (const auto& row : precomputedStringsRows){
        std::string hexString = assembleStringsRows(row);
        cv::Mat img = loadImageFromHexString(hexString, blockSize, blockSize);
        precomputedImages.push_back(img);
    }

    std::mutex mosaicMutex;

    time_t startTime = time(nullptr);
    std::cout << "Starting to generate the mosaic " << ctime(&startTime) << std::endl;

    auto processBlock = [&](int i, int j) {
        cv::Rect roi(j * blockSize, i * blockSize, blockSize, blockSize);
        cv::Mat block = inputImage(roi).clone();

        std::vector<std::string> inputBlocStringsRows = getImageRowsAsString(block);

        int **scoreTab = new int *[blockSize + 1];
        for (int x = 0; x <= blockSize; x++) {
            scoreTab[x] = new int[blockSize + 1];
            scoreTab[x][0] = x * -1;
        }

        int maxScore = std::numeric_limits<int>::min();
        cv::Mat *bestMatch = nullptr;

        for (auto& img : precomputedImages) {
            int score = alignmentScoreRowByRow(img, block, blockSize, scoreTab);
            if (score > maxScore) {
                maxScore = score;
                bestMatch = &img;
            }
        }

        for (int x = 0; x <= blockSize; x++) delete[] scoreTab[x];
        delete[] scoreTab;

        cv::Mat bestMatchImg = *bestMatch;
        {
            std::lock_guard<std::mutex> lock(mosaicMutex);
            bestMatchImg.copyTo(mosaic(roi));
        }
    };

    // Launch tasks in parallel
    std::vector<std::future<void>> futures;
    for (int i = 0; i < rowBlocks; ++i) {
        for (int j = 0; j < colBlocks; ++j) {
            futures.push_back(std::async(std::launch::async, processBlock, i, j));
        }
    }

    for (auto& f : futures) f.get(); // Wait for all

    time_t endTime = time(nullptr);

    std::cout << "Done in " << difftime(endTime, startTime) << " seconds" << std::endl;
    return mosaic;
}
