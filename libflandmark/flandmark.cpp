/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Mehmet Akif TAŞOVA
 * Copyright (C) 2014 Mehmet Akif TAŞOVA
 */

#include <iostream>
#include <cmath>

#include "flandmark.hpp"

 namespace flandmark 
 {
 	flandmark::flandmark(cv::Mat img, cv::Rect bbox, int offset_x0, int offset_y0, int offset_x1, int offset_y1)
 	: offset_x0(offset_x0), offset_y0(offset_y0), offset_x1(offset_x1), offset_y1(offset_y1), landmarksFound(false)
 	{
 		this->image = img;
 		this->bbox = bbox;
 		this->cascade.load(flandmark::DEFAULT_FACE_CASCADE_FILE);
 		this->flandmarkModel = flandmark_init(flandmark::DEFAULT_FLANDMARK_MODEL_FILE.c_str());
 		initLandmarks();
 	}	

	flandmark::flandmark(cv::Mat img, cv::Rect bbox, std::string flandmarkModelFile, int offset_x0, int offset_y0, int offset_x1, int offset_y1)
	: offset_x0(offset_x0), offset_y0(offset_y0), offset_x1(offset_x1), offset_y1(offset_y1), landmarksFound(false)
 	{
 		this->image = img;
 		this->bbox = bbox;
 		this->cascade.load(flandmark::DEFAULT_FACE_CASCADE_FILE);
 		this->flandmarkModel = flandmark_init(flandmarkModelFile.c_str());
 		initLandmarks();
 	}	


 	flandmark::~flandmark(){
 		freeLandmarks();
 		flandmark_free(this->flandmarkModel);
 	}


 	/**
 	 *
 	 *
 	 * returns cv::Rect objects in order of Right Eye Region, Left Eye Region and Mouth Region
 	 *
 	 */
	std::vector<cv::Rect> flandmark::getRegionOfInterets(void){
		std::vector<cv::Rect> data;

		data.push_back(this->getRightEyeRegion());
		data.push_back(this->getLeftEyeRegion());
		data.push_back(this->getMouthRegion());

		return data;

	}

	std::vector<cv::Point> flandmark::getLandmarks(void){
		std::vector<cv::Point> data;
		for(int i = 0; i < 7; ++i){
			data.push_back(getPointFromLandmarks(i));
		}
		return data;
	}

	cv::Point flandmark::getPointFromLandmarks(unsigned int pos){

		if(pos > 7){
			return cv::Point(0, 0);
		}

		unsigned int posX = 2 * pos;
		unsigned int posY = posX + 1;

		return cv::Point(int(this->landmarks[posX]), int(this->landmarks[posY]));
	}

	cv::Point flandmark::getFaceCenter(void){
		return this->getPointFromLandmarks(0);
	}

	cv::Point flandmark::getNoseCenter(void){
		return this->getPointFromLandmarks(7);
	}

	cv::Rect flandmark::getRightEyeRegion(void){
		return this->getRightEyeRegion(this->offset_x0, this->offset_y0, this->offset_x1, this->offset_y1);
	}

	cv::Rect flandmark::getLeftEyeRegion(void){
		return this->getLeftEyeRegion(this->offset_x0, this->offset_y0, this->offset_x1, this->offset_y1);
	}

	cv::Rect flandmark::getMouthRegion(void){
		return this->getMouthRegion(this->offset_x0, this->offset_y0, this->offset_x1, this->offset_y1);
	}


	cv::Rect flandmark::getRightEyeRegion(int offset_x0, int offset_y0, int offset_x1, int offset_y1){

		if(this->isLandmarksFound()){
			
			cv::Point rp = getPointFromLandmarks(5);
			cv::Point lp = getPointFromLandmarks(1);

			cv::Point point_up, point_down;

			point_up.x = rp.x + offset_x0;
			point_up.y = rp.y + offset_y0;

			point_down.x = lp.x + offset_x1;
			point_down.y = lp.y + offset_y1;

			return cv::Rect(point_up.x, point_up.y, std::abs(point_down.x - point_up.x), std::abs(point_down.y - point_up.y));
		}

		return flandmark::REGIONS_NOT_DETECTED;


	}

	cv::Rect flandmark::getLeftEyeRegion(int offset_x0, int offset_y0, int offset_x1, int offset_y1){

		if(this->isLandmarksFound()){
			
			cv::Point rp = getPointFromLandmarks(2);
			cv::Point lp = getPointFromLandmarks(6);

			cv::Point point_up, point_down;

			point_up.x = rp.x + offset_x0;
			point_up.y = rp.y + offset_y0;

			point_down.x = lp.x + offset_x1;
			point_down.y = lp.y + offset_y1;

			return cv::Rect(point_up.x, point_up.y, std::abs(point_down.x - point_up.x), std::abs(point_down.y - point_up.y));
		}

		return flandmark::REGIONS_NOT_DETECTED;

	}

	cv::Rect flandmark::getMouthRegion(int offset_x0, int offset_y0 , int offset_x1, int offset_y1 ){
		if(this->isLandmarksFound()){
			
			cv::Point rp = getPointFromLandmarks(3);
			cv::Point lp = getPointFromLandmarks(4);

			cv::Point point_up, point_down;

			point_up.x = rp.x + offset_x0;
			point_up.y = rp.y + offset_y0;

			point_down.x = lp.x + offset_x1;
			point_down.y = lp.y + offset_y1;

			return cv::Rect(point_up.x, point_up.y, std::abs(point_down.x - point_up.x), std::abs(point_down.y - point_up.y));
		}

		return flandmark::REGIONS_NOT_DETECTED;

	}

	bool flandmark::findLandmarks(void){

		if(false == (this->landmarksFound)){

			IplImage* iplImage = new IplImage(this->image);
			cv::Mat grayImage;

			cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
			cv::equalizeHist(grayImage, grayImage);

			IplImage* grayIplImage = new IplImage(grayImage);

			int *temp_bbox;
			this->rectToBbox(this->bbox, &temp_bbox);

			cv::rectangle(this->image, cv::Point(temp_bbox[0], temp_bbox[1]), cv::Point(temp_bbox[2], temp_bbox[3]), CV_RGB(0,0,255) );

			flandmark_detect(grayIplImage, temp_bbox, this->flandmarkModel, this->landmarks);

			this->landmarksFound = true;

			delete iplImage;
			delete grayIplImage;
			bboxFree(temp_bbox);
		}

		return (this->landmarksFound);
	}

	bool flandmark::isLandmarksFound(void) const{
		return (this->landmarksFound);
	}

	int* flandmark::bboxInit(void){
		return new int[4];
	}

	void flandmark::bboxFree(int *bbox){
		delete[] bbox;
	}

	/**
	 *	Connverts given cv::Rect object to bbox array and also initializes array too. please use
	 *	flandmark::bboxInit() and flandmark::bboxFree(int *bbox) methods to initialize and destroy bbox array
	 *	Also don't forget to free bbox array after you're done with it.
	 * \param[out] bbox
	 * \param[in] rect
	 */
	void flandmark::rectToBbox(cv::Rect rect, int **bbox){
		int *data = bboxInit();
		data[0] = rect.x;
		data[1] = rect.y;
		data[2] = (rect.x + rect.width);
		data[3] = (rect.y + rect.height);

		*bbox = data;
	}

	void flandmark::initLandmarks(void){
		this->landmarks = new double[2*((this->flandmarkModel->data).options.M)];
	}
	void flandmark::freeLandmarks(void){
		delete[] (this->landmarks);
	}

	const std::string flandmark::DEFAULT_FACE_CASCADE_FILE = "haarcascade_frontalface_alt.xml";
	const std::string flandmark::DEFAULT_FLANDMARK_MODEL_FILE = "flandmark_model.dat";
	const cv::Rect flandmark::REGIONS_NOT_DETECTED = cv::Rect(0, 0, 0, 0);
 }