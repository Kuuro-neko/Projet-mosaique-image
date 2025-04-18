#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem> 
#include <map>
#include <fstream>
#include <sstream>
#include <cmath>
#include <numeric>
#include <random>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <atomic>
#include "kmeans_mosaic.hpp"
#include "statistical_features.hpp"
#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_File_Browser.H>
#include <FL/Fl_File_Chooser.H>
#include <FL/Fl_Text_Buffer.H>
#include <FL/Fl_Text_Editor.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Text_Display.H>
#include <FL/Fl_Value_Input.H>
#include <FL/Fl_JPEG_Image.H>
#include <FL/Fl_PNG_Image.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Progress.H>
#include <FL/Fl_Choice.H>
Fl_Progress* progressBar;
Fl_Button* buttonMeanColor;
Fl_Button* buttonVariance;
Fl_Button* buttonSkewness ;
Fl_Button* buttonEnergy;
Fl_Button* buttonReuseImages ;
Fl_Button* buttonAlignement ;
Fl_Button* buttonKmeans ;
Fl_Button* buttonStatMoyenne ;
Fl_Choice* choice;
Fl_Value_Slider* kmeansClusterSlider;
Fl_Value_Slider* tailleSlider;
Fl_Value_Input *blocSize;
Fl_Text_Buffer *buffTE=new Fl_Text_Buffer();

#include "includes/alignmentMosaic.hpp"
#include "includes/meanFeatureMosaic.hpp"

#ifdef _WIN32
    std::string homeDir = getenv("USERPROFILE"); // Windows
#else
    std::string homeDir = getenv("HOME");       // Linux/Unix
#endif

using namespace cv;
namespace fs = std::filesystem;

struct MosaicParams {
    std::string bitArray;
    std::string datasetPath;
    int blockSize;
    std::string imagePath;
    Fl_Text_Buffer* textBuffer;
};
bool MC=true,V=true,S=false,E=false,IU=false,A=false,K=false,SM=true;
int kmeansCluster=400;
std::string image;
std::string datasetFolder;
std::string dossier;
int datasetSize=0;

void buttonParamImageMC(Fl_Widget* widget, void* data) {
    MC=!MC;
    if(MC){
        widget->color(FL_WHITE); 
        widget->labelcolor(FL_BLACK);
    }else{
        widget->color(FL_BLACK); 
        widget->labelcolor(FL_WHITE);
    }
}
void buttonParamImageV(Fl_Widget* widget, void* data) {
    V=!V;
    if(V){
        widget->color(FL_WHITE); 
        widget->labelcolor(FL_BLACK);
    }else{
        widget->color(FL_BLACK); 
        widget->labelcolor(FL_WHITE);
    }
}
void buttonParamImageS(Fl_Widget* widget, void* data) {
    S=!S;
    if(S){
        widget->color(FL_WHITE); 
        widget->labelcolor(FL_BLACK);
    }else{
        widget->color(FL_BLACK); 
        widget->labelcolor(FL_WHITE);
    }
}
void buttonParamImageE(Fl_Widget* widget, void* data) {
    E=!E;
    if(E){
        widget->color(FL_WHITE); 
        widget->labelcolor(FL_BLACK);
    }else{
        widget->color(FL_BLACK); 
        widget->labelcolor(FL_WHITE);
    }
}
void buttonParamImageIU(Fl_Widget* widget, void* data) {
    IU=!IU;
    if(IU){
        widget->color(FL_WHITE); 
        widget->labelcolor(FL_BLACK);
    }else{
        widget->color(FL_BLACK); 
        widget->labelcolor(FL_WHITE);
    }
}
void buttonParamImageAlignement(Fl_Widget* widget, void* data) {
    A=true;
    SM=false;
    K=false;
    buttonMeanColor->hide();
    buttonVariance->hide();
    buttonSkewness->hide();
    buttonEnergy->hide();
    buttonReuseImages->show();
    kmeansClusterSlider->hide();
    widget->color(FL_WHITE); 
    widget->labelcolor(FL_BLACK);
    buttonKmeans->color(FL_BLACK);
    buttonKmeans->labelcolor(FL_WHITE);
    buttonKmeans->redraw();
    buttonStatMoyenne->color(FL_BLACK);
    buttonStatMoyenne->labelcolor(FL_WHITE);
    buttonStatMoyenne->redraw();
    Fl::check();
}
void buttonParamImageStatMoyenne(Fl_Widget* widget, void* data) {
    SM=true;
    A=false;
    K=false;
    buttonMeanColor->show();
    buttonVariance->show();
    buttonSkewness->show();
    buttonEnergy->show();
    buttonReuseImages->show();
    kmeansClusterSlider->hide();
    widget->color(FL_WHITE); 
    widget->labelcolor(FL_BLACK);
    buttonKmeans->color(FL_BLACK);
    buttonKmeans->labelcolor(FL_WHITE);
    buttonKmeans->redraw();
    buttonAlignement->color(FL_BLACK);
    buttonAlignement->labelcolor(FL_WHITE);
    buttonAlignement->redraw();
    Fl::check();
}
void buttonParamImageKmeans(Fl_Widget* widget, void* data) {
    K=true;
    A=false;
    SM=false;
    buttonMeanColor->hide();
    buttonVariance->hide();
    buttonSkewness->hide();
    buttonEnergy->hide();
    buttonReuseImages->show();
    kmeansClusterSlider->show();
    widget->color(FL_WHITE); 
    widget->labelcolor(FL_BLACK);
    buttonStatMoyenne->color(FL_BLACK);
    buttonStatMoyenne->labelcolor(FL_WHITE);
    buttonStatMoyenne->redraw();
    buttonAlignement->color(FL_BLACK);
    buttonAlignement->labelcolor(FL_WHITE);
    buttonAlignement->redraw();
    Fl::check();
}
void kmeansClusterSliderCallback(Fl_Widget* widget, void* data) {
    Fl_Value_Slider* slider = (Fl_Value_Slider*)widget;
    kmeansCluster = (int)slider->value();
    std::cout << "Kmeans cluster value: " << kmeansCluster << std::endl;
}


char msg[250];
void fonctionButtonCreerImage(Fl_Widget* widget, void* data) {
    // progressBar->draw();
    MosaicParams* param = (MosaicParams*)data;
    std::string bitArray=std::to_string(MC) + std::to_string(V) + std::to_string(S) + std::to_string(E) + std::to_string(IU);
    GenerateMosaicParams params;
    Fl_Text_Buffer* textBuffer = param->textBuffer;
    std::string imageFinale="/";
    imageFinale += buffTE->text();
    imageFinale += choice->text();
    params.setFromBitArray(bitArray);
    params.toString();
    param->datasetPath = datasetFolder;
    param->blockSize = (int)blocSize->value();
    if (image == "") {
        fl_alert("Veuillez choisir une image !");
        return;
    }
    if (datasetFolder == "") {
        fl_alert("Veuillez choisir un dataset !");
        return;
    }
    cv::Mat inputImage = cv::imread(image, cv::IMREAD_COLOR);
    if (IU) {
        // compute the number of blocks
        int blockSize = ((MosaicParams*)data)->blockSize;
        int rowBlocks = inputImage.rows / blockSize;
        int colBlocks = inputImage.cols / blockSize;
        int totalBlocks = rowBlocks * colBlocks;
        if (totalBlocks == 0) {
            fl_alert("La taille du bloc est trop grande pour l'image !");
            return;
        }
        if (totalBlocks > datasetSize) {
            fl_alert("Le nombre de blocs est supérieur au nombre d'images dans le dataset ! Changez de dataset ou désactivez l'option 'Image unique'");
            return;
        }
    }
    std::map<std::string, StatisticalFeatures> meanValues = checkIfAlreadyPreProcessed(param->datasetPath);
    cv::Mat mosaic;
    if(K){
        mosaic=generateMosaicWithKMeans(inputImage, meanValues, param->blockSize, params, kmeansCluster, progressBar);
    }
    else if(A){
        if (IU)
            mosaic=generateMosaicUsingAlignmentSingleThread(inputImage, param->blockSize, param->datasetPath, IU);
        else
            mosaic=generateMosaicUsingAlignment(inputImage, param->blockSize, param->datasetPath);
    }else{
        mosaic=generateMosaic(inputImage, meanValues, param->blockSize, params, progressBar);
    }
    std::cout << " dossier : " << dossier << std::endl;
    std::cout << " imageFinale : " << imageFinale << std::endl;
    std::cout << " About to write image to file: " << dossier+imageFinale << std::endl;
    cv::imwrite(dossier+imageFinale, mosaic);
    float psnr = PSNR(inputImage, mosaic);
    sprintf(msg, "image mosaique PSNR : %f dB", psnr);
    Fl_JPEG_Image* image=new Fl_JPEG_Image((dossier+imageFinale).c_str());

    int maxW = 600;
    int maxH = 600;

    int ow = image->w();
    int oh = image->h();
    Fl_Image* to_display = image;
    if (ow > maxW || oh > maxH) {
        double scale = std::min((double)maxW / ow, (double)maxH / oh);
        int nw = (int)(ow * scale);
        int nh = (int)(oh * scale);

        Fl_RGB_Image* rgb = (Fl_RGB_Image*)image->copy(nw, nh);
        to_display = rgb;
        delete image;
    }

    Fl_Window* window=new Fl_Window(600,600);
    Fl_Box* box=new Fl_Box(0,0,600,600,msg);
    std::cout<<msg<<std::endl;
    box->image(to_display);
    window->end();
    window->show();
}

void fonctionChoisirImage(Fl_Widget* widget, void* data) {
    // Fl_Text_Buffer * param = (Fl_Text_Buffer *)data;
    std::map<Fl_Text_Buffer *,std::string> map = *(std::map<Fl_Text_Buffer *,std::string> *)data;
    Fl_Text_Buffer *param = map.begin()->first;
    // std::string image = map.begin()->second;
    std::string imageFolder = homeDir;
    imageFolder += "/Pictures";
    // test if imageFolder exists
    if (!fs::exists(imageFolder)) {
        imageFolder = homeDir;
        imageFolder += "/Images";
    }
    Fl_File_Chooser *fileChooser=new Fl_File_Chooser(imageFolder.c_str(), NULL, Fl_File_Chooser::SINGLE, "images");
    fileChooser->filter("Image Files (*.{jpeg,jpg,png})");
    fileChooser->show();
    while (fileChooser->shown()) {
        Fl::wait(); 
    }
    if (fileChooser->value() != NULL) {
        param->text(fileChooser->value());
        image = fileChooser->value();
        Fl_Image* original = nullptr;
        // Detect extension (lowercase comparison)
        std::string ext = image.substr(image.find_last_of('.') + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == "jpg" || ext == "jpeg") {
            original = new Fl_JPEG_Image(image.c_str());
        } else if (ext == "png") {
            original = new Fl_PNG_Image(image.c_str());
        }
        int ow=original->w();
        tailleSlider->bounds(1, ow);
        tailleSlider->value(16);
        tailleSlider->show();
        blocSize->show();

        std::string imageName = fs::path(image).stem().string(); // Get the base name without extension
        buffTE->text((imageName + "_mosaic").c_str());
    }else{
        param->text("");
        image = "";
    }
}
void fonctionChoisirDataset(Fl_Widget* widget, void* data) {
    std::map<Fl_Text_Buffer *,std::string> map = *(std::map<Fl_Text_Buffer *,std::string> *)data;
    Fl_Text_Buffer *param = map.begin()->first;
    Fl_File_Chooser *fileChooser=new Fl_File_Chooser(getenv("HOME"), NULL, Fl_File_Chooser::MULTI, "dataset");
    fileChooser->show();
    fileChooser->type(Fl_File_Chooser::DIRECTORY);
    while (fileChooser->shown()) {
        Fl::wait(); 
    }
    if (fileChooser->value() != NULL) {
        std::string datasetPath = fileChooser->value();
        param->text(datasetPath.c_str());
        datasetFolder = datasetPath;
    }else{
        param->text("");
        datasetFolder = "";
    }
    // get the number of files in the dataset
    datasetSize = std::distance(fs::directory_iterator(datasetFolder), fs::directory_iterator{});
    kmeansClusterSlider->bounds(1, datasetSize);
}
void fonctionChoisirDossier(Fl_Widget* widget, void* data) {
    std::map<Fl_Text_Buffer *,std::string> map = *(std::map<Fl_Text_Buffer *,std::string> *)data;
    Fl_Text_Buffer *param = map.begin()->first;
    Fl_File_Chooser *fileChooser=new Fl_File_Chooser(homeDir.c_str(), NULL, Fl_File_Chooser::MULTI, "dataset");
    fileChooser->show();
    fileChooser->type(Fl_File_Chooser::DIRECTORY);
    while (fileChooser->shown()) {
        Fl::wait(); 
    }
    if (fileChooser->value() != NULL) {
        std::string datasetPath = fileChooser->value();
        param->text(datasetPath.c_str());
        dossier = datasetPath;
    }else{
        param->text("");
        dossier = "";
    }
}
void fonctionVoirImage(Fl_Widget* widget, void* data) {
    if (image != "") {
        Fl_Image* original = nullptr;
        // Detect extension (lowercase comparison)
        std::string ext = image.substr(image.find_last_of('.') + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == "jpg" || ext == "jpeg") {
            original = new Fl_JPEG_Image(image.c_str());
        } else if (ext == "png") {
            original = new Fl_PNG_Image(image.c_str());
        } else {
            fl_alert("Format non supporté : %s", ext.c_str());
            return;
        }

        if (original->fail()) {
            fl_alert("Échec du chargement de l'image !");
            delete original;
            return;
        }

        int ow = original->w();
        int oh = original->h();

        int maxW = 600;
        int maxH = 600;

        Fl_Image* to_display = original;

        if (ow > maxW || oh > maxH) {
            double scale = std::min((double)maxW / ow, (double)maxH / oh);
            int nw = (int)(ow * scale);
            int nh = (int)(oh * scale);

            Fl_RGB_Image* rgb = (Fl_RGB_Image*)original->copy(nw, nh);
            to_display = rgb;
            delete original;
        }

        Fl_Window* window = new Fl_Window(600, 600, "Aperçu image");
        Fl_Box* box = new Fl_Box(0, 0, 600, 600);
        box->image(to_display);
        window->resizable(box);
        window->end();
        window->show();
    }
}

void displayExistingPrecomputedDataset(Fl_Text_Buffer *buff) {
    std::ifstream file(STATISTICAL_FEATURES_FILE);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << STATISTICAL_FEATURES_FILE << std::endl;
        return;
    }
    std::string line;
    std::getline(file, line);
    std::cout << "Dataset path: " << line << std::endl;
    datasetFolder = line;
    buff->text(line.c_str());

    datasetSize = std::distance(fs::directory_iterator(datasetFolder), fs::directory_iterator{});
    kmeansClusterSlider->bounds(1, datasetSize);
}

void fonctionSliderBloc(Fl_Widget* widget, void* data) {
    blocSize->value(tailleSlider->value());
    blocSize->redraw();
}

void fonctionInputBloc(Fl_Widget* widget, void* data) {
    if(blocSize->value()>tailleSlider->maximum()){
        blocSize->value(tailleSlider->maximum());
    }
    if(blocSize->value()<tailleSlider->minimum()){
        blocSize->value(tailleSlider->minimum());
    }
    tailleSlider->value(blocSize->value());
    tailleSlider->redraw();
}

int main(int argc, char** argv )
{
    Fl_Window* window = new Fl_Window(600, 600, "Interface mosaïque");
    progressBar=new Fl_Progress(0,575,600,25);
    Fl_Text_Buffer *buffText=new Fl_Text_Buffer();
    Fl_Text_Display *dispText = new Fl_Text_Display(0, 0, 600, 30);
    buffText->text("Caractéristiques de l'image :");
    dispText->buffer(buffText);
    dispText->color(FL_GRAY);
    buttonMeanColor = new Fl_Button(50, 50, 125, 25, "Couleur moyenne");
    buttonVariance = new Fl_Button(237, 50, 125, 25, "Variance");
    buttonSkewness = new Fl_Button(425, 50, 125, 25, "Asymétrie");
    buttonEnergy = new Fl_Button(50, 100, 125, 25, "Energie");
    buttonReuseImages = new Fl_Button(237, 100, 125, 25, "Image unique");
    buttonAlignement = new Fl_Button(50, 150, 125, 25, "Alignement");
    buttonKmeans = new Fl_Button(237, 150, 125, 25, "Kmeans");
    buttonStatMoyenne = new Fl_Button(425, 150, 125, 25, "Stat moyenne");
    kmeansClusterSlider = new Fl_Value_Slider(175, 50, 250, 25, "Kmeans clusters");
    buttonMeanColor->color(FL_WHITE); 
    buttonMeanColor->labelcolor(FL_BLACK);
    buttonMeanColor->callback(buttonParamImageMC);
    // buttonMeanColor->hide();
    buttonVariance->color(FL_WHITE); 
    buttonVariance->labelcolor(FL_BLACK);
    buttonVariance->callback(buttonParamImageV);
    buttonSkewness->color(FL_BLACK); 
    buttonSkewness->labelcolor(FL_WHITE);
    buttonSkewness->callback(buttonParamImageS);
    buttonEnergy->color(FL_BLACK); 
    buttonEnergy->labelcolor(FL_WHITE);
    buttonEnergy->callback(buttonParamImageE);
    buttonReuseImages->color(FL_BLACK); 
    buttonReuseImages->labelcolor(FL_WHITE);
    buttonReuseImages->callback(buttonParamImageIU);
    buttonAlignement->color(FL_BLACK); 
    buttonAlignement->labelcolor(FL_WHITE);
    buttonAlignement->callback(buttonParamImageAlignement);
    buttonKmeans->color(FL_BLACK); 
    buttonKmeans->labelcolor(FL_WHITE);
    buttonKmeans->callback(buttonParamImageKmeans);
    buttonStatMoyenne->color(FL_WHITE); 
    buttonStatMoyenne->labelcolor(FL_BLACK);
    buttonStatMoyenne->callback(buttonParamImageStatMoyenne);
    kmeansClusterSlider->hide();
    kmeansClusterSlider->type(FL_HOR_NICE_SLIDER);
    kmeansClusterSlider->bounds(1, 1000);
    kmeansClusterSlider->step(1);
    kmeansClusterSlider->value(400);
    Fl_Button* buttonChoisirDataset = new Fl_Button(10, 200, 110, 25, "Choisir dataset");
    Fl_Text_Buffer *buffdataset=new Fl_Text_Buffer();
    displayExistingPrecomputedDataset(buffdataset);
    Fl_Text_Display *dispdataset = new Fl_Text_Display(130, 200, 400, 50);
    std::map<Fl_Text_Buffer *,std::string> mapdataset{{buffdataset,datasetFolder}};
    buttonChoisirDataset->callback(fonctionChoisirDataset,&mapdataset);
    dispdataset->buffer(buffdataset);
    Fl_Button* buttonChoisirImage = new Fl_Button(10, 275, 100, 25, "Choisir image");
    Fl_Text_Buffer *buff=new Fl_Text_Buffer();
    Fl_Text_Display *disp = new Fl_Text_Display(120, 275, 400, 50);
    blocSize = new Fl_Value_Input(390, 350, 50, 30);
    blocSize->hide();
    blocSize->type(FL_INT_INPUT);
    blocSize->value(16);
    tailleSlider = new Fl_Value_Slider(140, 350, 250, 30);
    tailleSlider->hide();
    tailleSlider->type(FL_HOR_NICE_SLIDER);
    tailleSlider->step(1);
    tailleSlider->value(16);
    blocSize->callback(fonctionInputBloc);
    tailleSlider->callback(fonctionSliderBloc);
    Fl_Text_Buffer *buffTextBloc=new Fl_Text_Buffer();
    Fl_Text_Display *dispTextBloc = new Fl_Text_Display(10, 350, 125, 30);
    buffTextBloc->text("Taille des blocs :");
    dispTextBloc->buffer(buffTextBloc);
    dispTextBloc->color(FL_GRAY);
    // blocSize->value(16);
    
    Fl_Text_Editor *dispTE = new Fl_Text_Editor(10, 400, 400, 30, "Nom image de sortie");
    dispTE->buffer(buffTE);
    choice = new Fl_Choice(410, 400, 75, 30);
    choice->add(".jpg");
    choice->add(".jpeg");
    choice->add(".png");
    choice->value(0); 
    Fl_Button* buttonChoisirDossier = new Fl_Button(10, 450, 110, 25, "Choisir dossier");
    Fl_Text_Buffer *buffdossier=new Fl_Text_Buffer();
    Fl_Text_Display *dispdossier = new Fl_Text_Display(130, 450, 400, 50);
    dispdossier->buffer(buffdossier);
    std::map<Fl_Text_Buffer *,std::string> mapdossier{{buffdossier,dossier}};
    buttonChoisirDossier->callback(fonctionChoisirDossier,&mapdossier);
    std::map<Fl_Text_Buffer *,std::string> map{{buff,image}};
    buttonChoisirImage->callback(fonctionChoisirImage,&map);
    disp->buffer(buff);
    Fl_Button* buttonVoirImage = new Fl_Button(10, 300, 100, 25, "Voir image");
    buttonVoirImage->callback(fonctionVoirImage);
    Fl_Button* buttonCreerImage = new Fl_Button(250, 525, 100, 25, "Créer image");
    MosaicParams bitArrayMap{buffTE->text(),choice->text(),tailleSlider->value(),image,buffTE};
    buttonCreerImage->callback(fonctionButtonCreerImage,&bitArrayMap);
    // fileBrowser->load("/home/thibaut/Downloads/projet_image/Projet-mosaique-image/img");
    window->end();
    window->show();
    return Fl::run();
}
