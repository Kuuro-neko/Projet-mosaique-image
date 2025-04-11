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
#include <FL/Fl_Text_Display.H>
#include <FL/Fl_Value_Input.H>
#include <FL/Fl_JPEG_Image.H>
#include <FL/Fl_PNG_Image.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Progress.H>
Fl_Progress* progressBar;
Fl_Button* buttonMeanColor;
Fl_Button* buttonVariance;
Fl_Button* buttonSkewness ;
Fl_Button* buttonEnergy;
Fl_Button* buttonReuseImages ;
Fl_Button* buttonAlignement ;
Fl_Button* buttonKmeans ;
Fl_Button* buttonStatMoyenne ;

#include "includes/alignmentMosaic.hpp"
#include "includes/meanFeatureMosaic.hpp"

using namespace cv;
namespace fs = std::filesystem;

struct MosaicParams {
    std::string bitArray;
    std::string datasetPath;
    int blockSize;
    std::string imagePath;
};
bool MC=true,V=true,S=false,E=false,IU=false,A=false,K=false,SM=true;
std::string image;

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
    buttonReuseImages->hide();
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


char msg[250];
void fonctionButtonCreerImage(Fl_Widget* widget, void* data) {
    // progressBar->draw();
    MosaicParams* param = (MosaicParams*)data;
    std::string bitArray=std::to_string(MC) + std::to_string(V) + std::to_string(S) + std::to_string(E) + std::to_string(IU);
    GenerateMosaicParams params;
    params.setFromBitArray(bitArray);
    params.toString();
    std::map<std::string, StatisticalFeatures> meanValues = checkIfAlreadyPreProcessed(param->datasetPath);
    cv::Mat inputImage = cv::imread(image, cv::IMREAD_COLOR);
    cv::Mat mosaic;
    if(K){
        mosaic=generateMosaicWithKMeans(inputImage, meanValues, param->blockSize, params, 5);
    }
    else if(A){
        if (IU)
            mosaic=generateMosaicUsingAlignmentSingleThread(inputImage, param->blockSize, param->datasetPath, IU);
        else
            mosaic=generateMosaicUsingAlignment(inputImage, param->blockSize, param->datasetPath);
    }else{
        mosaic=generateMosaic(inputImage, meanValues, param->blockSize, params, progressBar);
    }
    cv::imwrite("mosaic_output_i.jpg", mosaic);
    float psnr = PSNR(inputImage, mosaic);
    sprintf(msg, "image mosaique PSNR : %f", psnr);
    Fl_JPEG_Image* image=new Fl_JPEG_Image("./mosaic_output_i.jpg");
    Fl_Window* window=new Fl_Window(600,600);
    Fl_Box* box=new Fl_Box(0,0,600,600,msg);
    std::cout<<msg<<std::endl;
    box->image(image);
    window->end();
    window->show();
}

void fonctionChoisirImage(Fl_Widget* widget, void* data) {
    // Fl_Text_Buffer * param = (Fl_Text_Buffer *)data;
    std::map<Fl_Text_Buffer *,std::string> map = *(std::map<Fl_Text_Buffer *,std::string> *)data;
    Fl_Text_Buffer *param = map.begin()->first;
    // std::string image = map.begin()->second;
    std::string imageFolder = getenv("HOME");
    imageFolder += "/Pictures";
    // test if imageFolder exists
    if (!fs::exists(imageFolder)) {
        imageFolder = getenv("HOME");
        imageFolder += "/Images";
    }
    Fl_File_Chooser *fileChooser=new Fl_File_Chooser(imageFolder.c_str(), NULL, Fl_File_Chooser::SINGLE, "images");
    fileChooser->show();
    while (fileChooser->shown()) {
        Fl::wait(); 
    }
    if (fileChooser->value() != NULL) {
        param->text(fileChooser->value());
        image = fileChooser->value();
    }else{
        param->text("");
        image = "";
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
        window->end();
        window->show();
    }
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
    buttonEnergy = new Fl_Button(50, 125, 125, 25, "Energie");
    buttonReuseImages = new Fl_Button(237, 125, 125, 25, "Image unique");
    buttonAlignement = new Fl_Button(50, 200, 125, 25, "Alignement");
    buttonKmeans = new Fl_Button(237, 200, 125, 25, "Kmeans");
    buttonStatMoyenne = new Fl_Button(425, 200, 125, 25, "Stat moyenne");
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
    Fl_Button* buttonChoisirImage = new Fl_Button(10, 300, 100, 25, "Choisir image");
    Fl_Text_Buffer *buff=new Fl_Text_Buffer();
    Fl_Text_Display *disp = new Fl_Text_Display(120, 300, 400, 50);
    Fl_Value_Input *blocSize = new Fl_Value_Input(125, 400, 50, 25, "Taille des blocs : ");
    blocSize->value(10);
    std::map<Fl_Text_Buffer *,std::string> map{{buff,image}};
    buttonChoisirImage->callback(fonctionChoisirImage,&map);
    disp->buffer(buff);
    Fl_Button* buttonVoirImage = new Fl_Button(10, 325, 100, 25, "Voir image");
    buttonVoirImage->callback(fonctionVoirImage);
    Fl_Button* buttonCreerImage = new Fl_Button(250, 500, 100, 25, "Créer image");
    MosaicParams bitArrayMap{"bitArray",argv[1],blocSize->value(),image};
    buttonCreerImage->callback(fonctionButtonCreerImage,&bitArrayMap);
    // fileBrowser->load("/home/thibaut/Downloads/projet_image/Projet-mosaique-image/img");
    window->end();
    window->show();
    return Fl::run();
}
