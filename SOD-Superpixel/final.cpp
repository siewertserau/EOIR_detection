#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>

#include "SLIC.h"
#include <cstdlib>
#include <cstdio>


#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <vector>
#include <time.h>
#include <numeric>

using namespace cv;
using namespace std;

int* finalLabels;
int flowMapRows = 0;
int flowMapCols = 0;
std::vector<float> qmvX;
std::vector<float> qmvY;
int countInMagBins[10];
Mat frame;
// number of bins for the histogram
int numBins  = 10;

struct timespec start_time;
struct timespec end_time;

// Function used to calculate the median of all the values in a vector
double median(vector<float> scores)
{
  double median;
  size_t size = scores.size();

  sort(scores.begin(), scores.end());

  if (size  % 2 == 0)
  {
      median = (scores[size / 2 - 1] + scores[size / 2]) / 2;
  }
  else 
  {
      median = scores[size / 2];
  }

  return median;
}

// Function used to draw the optical flow of each point on the image
static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
                    double, const Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

// function to find the maximum value in a vector 
double max_find(vector<float> x)
{
  double max = 0;
  for (int i = 0; i < x.size(); i++)
  {
    if (max < x.at(i)) 
      max = x.at(i);
  }
  return max;
}

// function used to calculate the frame histogram
void frameHist(const Mat& flow)
{
    // varaibles used to split the image to magnitude and angle
  // matrices to store the magnitude and angle of the flow vectors
  Mat magnitude,angle;
  // splitting the value of the flow vector to magnitude and angle
  vector<Mat> planes;
  split( flow, planes );
  cartToPolar(planes[0],planes[1], magnitude, angle, true);


  std::vector<float> magBins[10];
  std::vector<float> angBins[10];

  // Maximum value for histogram bins
  double max=100;

  // variables used for iteration
  int u=0,v=0;
  // temporary variables for intermediate calculation
  float temp=0;

  // dividing the image into different bins
  for(u=0;u<flowMapRows;u++)
  {
    for(v=0;v<flowMapCols;v++)
    {
      temp = magnitude.at<float>(u, v);
      if(temp < max/numBins)
      {
        magBins[0].push_back(temp);
        angBins[0].push_back(angle.at<float>(u, v));
      }
      else if(temp< (max*2)/numBins)
      {
        magBins[1].push_back(temp);
        angBins[1].push_back(angle.at<float>(u, v));
      }
      else if(temp< (max*3)/numBins)
      {
        magBins[2].push_back(temp);
        angBins[2].push_back(angle.at<float>(u, v));
      }
      else if(temp< (max*4)/numBins)
      {
        magBins[3].push_back(temp);
        angBins[3].push_back(angle.at<float>(u, v));
      }
      else if(temp< (max*5)/numBins)
      {
        magBins[4].push_back(temp);
        angBins[4].push_back(angle.at<float>(u, v));
      }
      else if(temp< (max*6)/numBins)
      {
        magBins[5].push_back(temp);
        angBins[5].push_back(angle.at<float>(u, v));
      }
      else if(temp< (max*7)/numBins)
      {
        magBins[6].push_back(temp);
        angBins[6].push_back(angle.at<float>(u, v));
      }
      else if(temp< (max*8)/numBins)
      {
        magBins[7].push_back(temp);
        angBins[7].push_back(angle.at<float>(u, v));
      }
      else if(temp< (max*9)/numBins)
      {
        magBins[8].push_back(temp);
        angBins[8].push_back(angle.at<float>(u, v));
      }
      else if(temp< (max))
      {
        magBins[9].push_back(temp);
        angBins[9].push_back(angle.at<float>(u, v));
      }
    }
  }

  // calculating the quantized motion vector for each bin
  for(u=0;u<numBins;u++)
  {
    std::vector<float> xAxis;
    std::vector<float> yAxis;
    if((int)magBins[u].size() > 0)
    {
      polarToCart(magBins[u],angBins[u],xAxis,yAxis,true);
      temp = std::accumulate(xAxis.begin(),xAxis.end(),0);
      countInMagBins[u] = (int)magBins[u].size();
      qmvX.push_back((float)(temp/countInMagBins[u]));
      temp = std::accumulate(yAxis.begin(),yAxis.end(),0);
      qmvY.push_back((float)(temp/(int)angBins[u].size()));
    }
    else // case if there are no element in the bin
    {
      qmvX.push_back(0);
      qmvY.push_back(0);
    }
  }
}

// calculating the superpixel level histogram of the image
void superpixelHist(const Mat& flow)
{
  // varaibles used to split the image to magnitude and angle
  // matrices to store the magnitude and angle of the flow vectors
  Mat magnitude,angle;
  // splitting the value of the flow vector to magnitude and angle
  vector<Mat> planes;
  split( flow, planes );
  cartToPolar(planes[0],planes[1], magnitude, angle, true);
  
  Mat spMag,motionSig;
  magnitude.copyTo(spMag);
  magnitude.copyTo(motionSig);
  vector<float> sp[204];
  std::vector<float> mhSP;

  int countInSpMagBins[204][10] = {0};

  double max=100;
  double temp=0;

  // variables used for iteration
  int t=0,u=0,v=0;
  int n = 0;

  // storing each pixel into its corresponding superpixel vector
  for(u=0;u<flowMapRows;u++)
  {
    for(v=0;v<flowMapCols;v++)
    {
      sp[(int)(finalLabels[(u*flowMapCols)+v])].push_back(magnitude.at<float>(u, v));
    }
  }

  // used to identify the median of all values in a superpixel
  for(t=0;t<finalNumSuperpixels;t++)
  {
    temp = median(sp[t]);
    n = sp[t].size();
    mhSP.push_back(temp);
  }

  // dividing the superpixels into bins
  for(u=0;u<finalNumSuperpixels;u++)
  {
    for(v=0;v<(int)sp[u].size();v++)
    {
      temp = sp[u][v];
      if(temp < max/numBins)
        countInSpMagBins[u][0]++;
      else if(temp< (max*2)/numBins)
        countInSpMagBins[u][1]++;
      else if(temp< (max*3)/numBins)
        countInSpMagBins[u][2]++;
      else if(temp< (max*4)/numBins)
        countInSpMagBins[u][3]++;
      else if(temp< (max*5)/numBins)
        countInSpMagBins[u][4]++;
      else if(temp< (max*6)/numBins)
        countInSpMagBins[u][5]++;
      else if(temp< (max*7)/numBins)
        countInSpMagBins[u][6]++;
      else if(temp< (max*8)/numBins)
        countInSpMagBins[u][7]++;
      else if(temp< (max*9)/numBins)
        countInSpMagBins[u][8]++;
      else if(temp< (max))
        countInSpMagBins[u][9]++;
    }
  }

  double temp1=0, temp2 =0;
  std::vector<float> SMD;
  std::vector<float> MSt;

  // calculating the motion distinctiveness of each superpixel
  for(t=0;t<finalNumSuperpixels;t++)
  {
    temp2 = 0;
    for(u=0;u<numBins;u++)
    {
      temp1 = 0;
      for(v=0;v<numBins;v++)
      {
        temp = sqrt(pow((qmvX[u]-qmvX[v]),2) + pow((qmvY[u]-qmvY[v]),2));
        temp*= countInMagBins[v];
        temp1+= temp;
      }
      temp2+= (temp1 * countInSpMagBins[t][u]);
    }
    SMD.push_back(temp2);
    MSt.push_back(temp2*(int)sp[t].size());
    // printf("%lf ", temp2);
  }

  // storing the median value of each superpixel into all elements of the image
  for(t=0;t<mhSP.size();t++)
  {
    for(u=0;u<flowMapRows;u++)
    {
      for(v=0;v<flowMapCols;v++)
      {
        if(finalLabels[(u*flowMapCols)+v] == t)
        {
          spMag.at<float>(u, v) = mhSP[t];
          motionSig.at<float>(u, v) = MSt[t];
        }
      }
    } 
  }

  double maxVal, minVal;
  cv::minMaxLoc(spMag, &minVal, &maxVal);

  imshow("output",spMag);

  // applying threshold
  spMag.convertTo(spMag,CV_8UC1,1);
  threshold(spMag,spMag,(maxVal*1)/4,255,THRESH_BINARY);
  imshow("thresholded image",spMag);
}

// function to caculate the Color histogram of the image
void colorHist(const Mat& ft)
{
  String strn = "Color Histogram";

  /// Separate the image in 3 places ( B, G and R )
  vector<Mat> bgr_planes;
  split( ft, bgr_planes );

  /// Establish the number of bins
  int histSize = 16;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256} ;
  const float* histRange = { range };
  bool uniform = true; bool accumulate = false;

  Mat b_hist, g_hist, r_hist;

  /// Compute the histograms:
  calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

  // Draw the histograms for B, G and R
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 255,255,255) );
  normalize(b_hist, b_hist, 1, histImage.rows, NORM_L2, -1, Mat() );
  normalize(g_hist, g_hist, 1, histImage.rows, NORM_L2, -1, Mat() );
  normalize(r_hist, r_hist, 1, histImage.rows, NORM_L2, -1, Mat() );

  /// Draw for each channel
  for( int i = 1; i < histSize; i++ )
  {
/*    rectangle( histImage, Point(i*bin_w, hist_h),
                   Point((i+1)*bin_w, hist_h - cvRound(b_hist.at<float>(i))),
                   Scalar( 255, 0, 0), 1, 8, 0 );
    rectangle( histImage, Point(i*bin_w, hist_h),
                   Point((i+1)*bin_w, hist_h - cvRound(g_hist.at<float>(i))),
                   Scalar( 0, 255, 0), 1, 8, 0 );
    rectangle( histImage, Point(i*bin_w, hist_h),
                   Point((i+1)*bin_w, hist_h - cvRound(r_hist.at<float>(i))),
                   Scalar( 0, 0, 255), 1, 8, 0 );*/
    line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
    line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                       Scalar( 0, 255, 0), 2, 8, 0  );
    line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                       Scalar( 0, 0, 255), 2, 8, 0  );
  }

  /// Display
  namedWindow(strn, 0 );
  imshow(strn, histImage );
}

int main(int argc, char** argv)
{
  // number of superpixels the image should be divided into
  int numSuperpixel;
  finalNumSuperpixels = 200;
  if (argc != 2) 
  {
    printf("number of superpixels not specified. assuming default value as 200\n");
    numSuperpixel = 200; // default value
  }
  else
    numSuperpixel = atoi(argv[1]);

  // object of class SLIC used to create superpixels
  SLIC slic;

  // capture video from camera 0
  VideoCapture cap(0);
  // setting the frame width and height
  cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
  cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

  if( !cap.isOpened() )
      return -1;

  // variables used to determine the optical flow
  Mat flow, cflow;
  UMat gray, prevgray, uflow;
  namedWindow("flow", 1);

  while(1)
  {
    // reading a frame
    cap >> frame;
    // converting the frame to grayscale
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    if( !prevgray.empty() )
    {
      clock_gettime(CLOCK_REALTIME, &start_time);
      // generating superpixels
      slic.GenerateSuperpixels(frame, numSuperpixel);
      // obtaining the label for each pixel in the frame
      finalLabels = slic.GetLabel();
/*      // draw the contours of the superpixels on the image
      if (frame.channels() == 3){ 
        Mat result = slic.GetImgWithContours(Scalar(255, 0, 0));
        imshow("segmented image",result);
      }
*/
      // calculating the optical flow of the two adjacent images
      calcOpticalFlowFarneback(prevgray, gray, uflow, 0.5, 3, 15, 3, 5, 1.2, OPTFLOW_FARNEBACK_GAUSSIAN);
      // converting the image from gray scale to rgb
      cvtColor(prevgray, cflow, COLOR_GRAY2RGB);
      uflow.copyTo(flow);
      flowMapRows = flow.rows;
      flowMapCols = flow.cols;
      // calculating the frame Histogram of the image
      frameHist(flow);
      // calculating the superpixel histogram of the image
      superpixelHist(flow);
      clock_gettime(CLOCK_REALTIME, &end_time);
      double framedt=0;
      framedt=(((end_time.tv_sec - start_time.tv_sec)*1000)+ ((end_time.tv_nsec-start_time.tv_nsec)/1000000));
      printf("\nTime taken saliency detection is %lf", framedt);
      
      imshow("flow", cflow);
    }
    if(waitKey(30)==27)
      break;

    // swapping the contents of the matrices
    std::swap(prevgray, gray);
  }
  return 0;
}
