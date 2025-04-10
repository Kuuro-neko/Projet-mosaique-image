#include "statistical_features.hpp"
#include <string>

GenerateMosaicParams::GenerateMosaicParams(bool meanColor, bool variance, bool skewness, bool energy, bool reuseImages)
    : meanColor(meanColor), variance(variance), skewness(skewness), energy(energy), reuseImages(reuseImages) {}

/**
 * @brief Définir les paramètres à partir d'une chaîne de caractères
 * 
 * @param bitArray Chaîne de caractères représentant les paramètres
 */
void GenerateMosaicParams::setFromBitArray(const std::string& bitArray) {
    if (bitArray.size() >= 5) {
        meanColor = bitArray[0] == '1';
        variance = bitArray[1] == '1';
        skewness = bitArray[2] == '1';
        energy = bitArray[3] == '1';
        reuseImages = bitArray[4] == '1';
    }
}

/**
 * @brief Convertir les paramètres en chaîne de caractères
 * 
 * @return std::string
 */
std::string GenerateMosaicParams::toString() const {
    std::string result = "meanColor : " + std::to_string(meanColor) + ", ";
    result += "variance : " + std::to_string(variance) + ", ";
    result += "skewness : " + std::to_string(skewness) + ", ";
    result += "energy : " + std::to_string(energy) + ", ";
    result += "reuseImages : " + std::to_string(reuseImages);
    return result;
}
