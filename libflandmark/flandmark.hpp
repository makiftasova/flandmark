/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Mehmet Akif TAŞOVA
 * Copyright (C) 2014 Mehmet Akif TAŞOVA
 */

#include <vector>
#include <string>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "flandmark_detector.h"

#ifndef __FLANDMARK__H_
#define __FLANDMARK__H_

namespace flandmark
{
	class flandmark
	{
	public:
		flandmark(cv::Mat img, cv::Rect bbox, int offset_x0 = -5, int offset_y0 = -20, int offset_x1 = +5, int offset_y1 = +20);
		flandmark(cv::Mat img, cv::Rect bbox, const std::string flandmarkModelFile, int offset_x0 = -5, int offset_y0 = -20, int offset_x1 = +5, int offset_y1 = +20);
		virtual ~flandmark();

		std::vector<cv::Rect> getRegionOfInterets(void);
		std::vector<cv::Point> getLandmarks(void);

		cv::Point getPointFromLandmarks(unsigned int pos);

		cv::Point getFaceCenter(void);
		cv::Point getNoseCenter(void);

		cv::Rect getRightEyeRegion(void);
		cv::Rect getLeftEyeRegion(void);
		cv::Rect getMouthRegion(void);

		cv::Rect getRightEyeRegion(int offset_x0, int offset_y0, int offset_x1, int offset_y1);
		cv::Rect getLeftEyeRegion(int offset_x0, int offset_y0, int offset_x1, int offset_y1);
		cv::Rect getMouthRegion(int offset_x0, int offset_y0, int offset_x1, int offset_y1);

		bool findLandmarks(void);
		bool isLandmarksFound(void) const;

		static const std::string DEFAULT_FACE_CASCADE_FILE;
		static const std::string DEFAULT_FLANDMARK_MODEL_FILE;

		static const cv::Rect REGIONS_NOT_DETECTED;

	private:
		// offset for left hand side points
		int offset_x0; 
		int offset_y0;
		// offset for right hand side points
		int offset_x1;
		int offset_y1;

		cv::CascadeClassifier cascade;
		cv::Mat image;
		cv::Rect bbox;
		FLANDMARK_Model * flandmarkModel;
		double *landmarks;
		bool landmarksFound;


		void rectToBbox(cv::Rect rect, int **bbox);

		int* bboxInit();
		void bboxFree(int *bbox);

		void initLandmarks();
		void freeLandmarks();
		
	};
}

#endif