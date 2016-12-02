// Copyright (c) 2008, Sebastian Montabone
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the owner nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
// 
// File:   saliencyFineGrained.h
// Author: Sebastian Montabone <samontab at puc.cl>
//
// Created on September 24, 2008, 2:57 PM
// Updated on July 6, 2015, 5:29 PM
//
// Fine-grained Saliency library (FGS). 
// http://www.samontab.com/web/saliency/
// This library calculates fine grained saliency in real time using integral images.
// It requires OpenCV.
//

#ifndef _saliencyFineGrained_H
#define	_saliencyFineGrained_H

#include <cstdio>
#include <string>
#include <iostream>
#include <stdint.h>
#include <opencv2/opencv.hpp>
 
/**
 * \brief Fine grained saliency (FGS)
 * 
 * Fine grained saliency based on algorithms described in [1]
 * [1] Montabone, Sebastian and Soto, Alvaro. "Human Detection Using a Mobile Platform and Novel Features Derived from a Visual Saliency Mechanism." Image and Vision Computing, 2010.
 *
 * Source code from http://www.samontab.com/web/saliency/
 */
class StaticSaliencyFineGrained
{
public:

  StaticSaliencyFineGrained();
  virtual ~StaticSaliencyFineGrained();
  bool computeSaliencyImpl( cv::Mat image, cv::Mat &saliencyMap );

private:
  void calcIntensityChannel(cv::Mat src, cv::Mat dst);
  void copyImage(cv::Mat src, cv::Mat dst);
  void getIntensityScaled(cv::Mat integralImage, cv::Mat gray, cv::Mat saliencyOn, cv::Mat saliencyOff, int neighborhood);
  float getMean(cv::Mat srcArg, cv::Point2i PixArg, int neighbourhood, int centerVal);
  void mixScales(cv::Mat *saliencyOn, cv::Mat intensityOn, cv::Mat *saliencyOff, cv::Mat intensityOff, const int numScales);
  void mixOnOff(cv::Mat intensityOn, cv::Mat intensityOff, cv::Mat intensity);
  void getIntensity(cv::Mat srcArg, cv::Mat dstArg,  cv::Mat dstOnArg,  cv::Mat dstOffArg, bool generateOnOff);
};

#endif	/* _saliencyFineGrained_H */

