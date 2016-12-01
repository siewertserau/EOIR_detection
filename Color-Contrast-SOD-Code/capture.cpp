//
//  capture.cpp
//  ECEN 5840-906
//  Created by AKSHAY SINGH
//  For Independent study under Dr. Sam Siewert
//  This code takes in an input image and produces a saliency map for the image
//  Code reference: http://mmcheng.net/code-data/

#include "capture.hpp"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

typedef std::pair<float, int> CostfIdx;
typedef std::vector<double> vecD;
struct timespec start,stop;

int main(int argc, char* argv[])
{
	if (argc!=2)
	{
		printf("Please enter image file name");
		return(0);
	}
	char *name (argv[1]);
    cvNamedWindow("Window", CV_WINDOW_AUTOSIZE);
    cv::Mat img=cv::imread(name);   //load image
    clock_gettime(CLOCK_REALTIME, &start);  //get start time
    img.convertTo(img, CV_32FC3, 1.0/255);
    
    //Quantize image to 12 colors in each channel
    cv::Mat id;
    double drp=0.95;
    const int clrNums[3] = {12, 12, 12};
    float clrTmp[3] = {clrNums[0] - 0.0001f, clrNums[1] - 0.0001f, clrNums[2] - 0.0001f};
    int w[3] = {clrNums[1] * clrNums[2], clrNums[2], 1};
    id = cv::Mat::zeros(img.size(), CV_32S);
    int rows = img.rows, cols = img.cols;
    if (img.isContinuous() && id.isContinuous()){
        cols *= rows;
        rows = 1;
    }
    
    // Build color pallete
    std::map<int, int> pallete;
    for (int y = 0; y < rows; y++)
    {
        const float* imgData = img.ptr<float>(y);
        int* idx = id.ptr<int>(y);
        for (int x = 0; x < cols; x++, imgData += 3)
        {
            idx[x] = (int)(imgData[0]*clrTmp[0])*w[0] + (int)(imgData[1]*clrTmp[1])*w[1] + (int)(imgData[2]*clrTmp[2]);
            pallete[idx[x]] ++;
        }
    }
    
    // Find significant colors
    int maxNum=0;
    int count = 0;
    std::vector<std::pair<int, int> > num; // (num, color) pairs in num
    num.reserve(pallete.size());
    
    for (std::map<int, int>::iterator it = pallete.begin(); it != pallete.end(); it++)
        num.push_back(std::pair<int, int>(it->second, it->first)); // (color, num) pairs in pallete
    
    sort(num.begin(), num.end(), std::greater<std::pair<int, int> >());   //sort histogram
    maxNum = (int)num.size();
    
    int maxDropNum = cvRound(rows * cols * (1-drp));    //calculate max no of colors that can be dropped
    for (int crnt = num[maxNum-1].first; crnt < maxDropNum && maxNum > 1; maxNum--)
        crnt += num[maxNum - 2].first;
    maxNum = fmin(maxNum, 256);
    if (maxNum <= 10)
        maxNum = fmin(10, (int)num.size());
    pallete.clear();
    for (int i = 0; i < maxNum; i++)
        pallete[num[i].second] = i;

    std::vector<cv::Vec3i> colormat(num.size());
    for (unsigned int i = 0; i < num.size(); i++)
    {
        colormat[i][0] = num[i].second / w[0];
        colormat[i][1] = num[i].second % w[0] / w[1];
        colormat[i][2] = num[i].second % w[1];
    }
    
    for (unsigned int i = maxNum; i < num.size(); i++)
    {
        int simIdx = 0, simVal = INT_MAX;
        for (int j = 0; j < maxNum; j++)
        {
            int d_ij;
            for (int k=0; k<3; k++) d_ij += pow((colormat[i][k] - colormat[j][k]),2);
            if (d_ij < simVal)
                simVal = d_ij, simIdx = j;
        }
        pallete[num[i].second] = pallete[num[simIdx].second];
    }
    cv::Mat _color3f = cv::Mat::zeros(1, maxNum, CV_32FC3);
    cv::Mat _colorNum = cv::Mat::zeros(_color3f.size(), CV_32S);
    
    cv::Vec3b* color = (cv::Vec3b*)(_color3f.data);
    
    int* colorNum = (int*)(_colorNum.data);
    for (int y = 0; y < rows; y++)
    {
        const cv::Vec3b* imgData = img.ptr<cv::Vec3b>(y);
        int* idx = id.ptr<int>(y);
        for (int x = 0; x < cols; x++)
        {
            idx[x] = pallete[idx[x]];
            color[idx[x]] += imgData[x];
            colorNum[idx[x]] ++;
        }
    }
    for (int i = 0; i < _color3f.cols; i++)
        color[i] /= (float)colorNum[i];

    //quantization done
    cv::cvtColor(_color3f, _color3f, CV_BGR2Lab);   //convert to L*a*b* color space
    
    cv::Mat wt;
    normalize(_colorNum, wt, 1, 0, cv::NORM_L1,CV_32F);
    int binN = _color3f.cols;
    cv::Mat _colorSal = cv::Mat::zeros(1, binN, CV_32F);
    float* colorSal = (float*)(_colorSal.data);
    std::vector<std::vector<CostfIdx> > similar(binN); // Get how similar the colors are and thier index
    cv::Vec3b* color1 = (cv::Vec3b*)(_color3f.data);
    float *w1 = (float*)(wt.data);
    for (int i = 0; i < binN; i++){
        std::vector<CostfIdx> &similari = similar[i];
        similari.push_back(std::make_pair(0.f, i));
        for (int j = 0; j < binN; j++){
            if (i == j)
                continue;
            int d_ij1;
            for (int k=0; k<3; k++) d_ij1 += pow((color1[i][k] - color1[j][k]),2);  //Calculate color distance metric
            d_ij1 = sqrt(d_ij1);
            similari.push_back(std::make_pair(d_ij1, j));
            colorSal[i] += w[j] * d_ij1;
        }
        sort(similari.begin(), similari.end());     //sort histogram
    }
    
    //color space smoothing
    cv::Mat colorNum1i = cv::Mat::ones(_colorSal.size(), CV_32SC1);
    float delta=0.25f;
    if (_colorSal.cols < 2)
        return(0);
    CV_Assert(_colorSal.rows == 1 && _colorSal.type() == CV_32FC1);
    CV_Assert(colorNum1i.size() == _colorSal.size() && colorNum1i.type() == CV_32SC1);
    binN = _colorSal.cols;
    cv::Mat newSal1d= cv::Mat::zeros(1, binN, CV_64FC1);
    float *sal = (float*)(_colorSal.data);
    double *newSal = (double*)(newSal1d.data);
    int *pW = (int*)(colorNum1i.data);
    
    // Distance based smooth
    int n = fmax(cvRound(binN * delta), 2);
    vecD dist(n, 0), val(n), w2(n);
    for (int i = 0; i < binN; i++){
        const std::vector<CostfIdx> &similari = similar[i];
        double totalDist = 0, tWt = 0;
        for (int j = 0; j < n; j++){
            int ithIdx =similari[j].second;
            dist[j] = similari[j].first;
            val[j] = sal[ithIdx];
            w2[j] = pW[ithIdx];
            totalDist += dist[j];
            tWt += w2[j];
        }
        double valCrnt = 0;
        for (int j = 0; j < n; j++)
            valCrnt += val[j] * (totalDist - dist[j]) * w2[j];
        
        newSal[i] =  valCrnt / (totalDist * tWt);
    }
    normalize(newSal1d, _colorSal, 0, 1, cv::NORM_MINMAX, CV_32FC1);
    
    colorSal = (float*)(_colorSal.data);
    cv::Mat salHC1f(img.size(), CV_32F);
    for (int r = 0; r < img.rows; r++){
        float* salV = salHC1f.ptr<float>(r);
        int* _idx = id.ptr<int>(r);
        for (int c = 0; c < img.cols; c++)
            salV[c] = colorSal[_idx[c]];
    }
    GaussianBlur(salHC1f, salHC1f, cv::Size(3, 3), 0);
    normalize(salHC1f, salHC1f, 0, 1, cv::NORM_MINMAX);
    clock_gettime(CLOCK_REALTIME, &stop);
    printf("Time taken:%ld sec %ld nSec \n", stop.tv_sec-start.tv_sec,stop.tv_nsec-start.tv_nsec);
    std::string ch="sal_";
	ch.append(name);
    while(1)
    {
        cv::imshow("Window",salHC1f);
        char c = cv::waitKey(0);
        if (c==27) break;
    }
	cv::imwrite(ch ,salHC1f);   //save output saliency map
    return(0);
}


