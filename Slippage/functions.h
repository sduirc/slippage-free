#pragma once

#include "imgset.h"
#include"dataset.h"


int getKltkeyPoints(Mat img, vector<Point2f> &points, int nmax, Mat &mask=Mat() );

cv::Mat findHomographyEx(const std::vector<Point2f> &prePoints, const std::vector<Point2f> &curPoints, int method=CV_RANSAC, double ransacThreshold=3.0, std::vector<Point2f> *preGood=NULL, std::vector<Point2f> *curGood=NULL);

int getKltTranslation( cv::Mat preImg, cv::Mat curImg,  const vector<Point2f> &preKeyPoints, 
				   vector<Point2f> &keyPoints, cv::Mat &similarity=Mat(), vector<Point2f> *preGood=NULL, vector<Point2f> *curGood=NULL );

//int KLTTrack( cv::Mat preImg, cv::Mat curImg, const vector<Point2f> &preKeyPoints, 
//				   vector<Point2f> &preMatch, vector<Point2f> &curMatch);

int KLTTrackNoRansac( cv::Mat preImg, cv::Mat curImg, const vector<Point2f> &preKeyPoints, 
				   vector<Point2f> &preMatch, vector<Point2f> &curMatch, int maxLevel);

int getSiftKeyPoints( Mat img, vector<KeyPoint> &keyPoints, Mat &description, Mat &mask=Mat() );


int getSiftMatchCount(Mat descriptor1, Mat descriptor2, const vector<KeyPoint> &keyPoints1, 
				 const vector<KeyPoint> &keyPoints2, int &goodMatchCount, const Mat &mask=Mat());

//float _get_distortion_error(const cv::Mat &T, const std::vector<cv::Point3f> &pt);

//float _get_distortion_error(const cv::Mat &T, const std::vector<cv::Point2f> &pt);

float _get_transform_error(const cv::Mat &T, std::vector<cv::Point3f> &pt, const std::vector<cv::Point3f> &ref_pt);

float _get_transform_error(const cv::Mat &T, std::vector<cv::Point2f> &pt, const std::vector<cv::Point2f> &ref_pt);

float estimateShift(const std::vector<Point2f> &pti, const std::vector<Point2f> &ptj);

//float getMatchError(const Mat &descriptor1, const Mat &descriptor2, const vector<Point2f> &keyPoints1, 
//				 const vector<Point2f> &keyPoints2, const std::vector<cv::Point3f> &pt, const Mat &mask=Mat());


int getSiftMatch(const Mat &descriptor1, const Mat &descriptor2, const vector<Point2f> &keyPoints1, const vector<Point2f> &keyPoints2, 
				 vector<Point2f> &goodPoints1, vector<Point2f> &goodPoints2, const Mat &mask=Mat());

int getSiftMatchNoRansac(const Mat &descriptor1, const Mat &descriptor2, const vector<Point2f> &keyPoints1, 
				 const vector<Point2f> &keyPoints2, vector<Point2f> &goodPoints1, vector<Point2f> &goodPoints2, const Mat &mask=Mat());

int getSiftMatch(const Mat &descriptor1, const Mat &descriptor2, const vector<Point2f> &keyPoints1, const vector<Point2f> &keyPoints2, 
				 vector<Point2f> &goodPoints1, vector<Point2f> &goodPoints2, vector<BgFrameData::MyMatch> &myMatches, const Mat &mask=Mat());

int getSiftMatchTopN(const Mat &descriptor1, const Mat &descriptor2, const vector<KeyPoint> &keyPoints1, const vector<KeyPoint> &keyPoints2, 
					 vector<Point2f> &goodPoints1, vector<Point2f> &goodPoints2, const int &n, const Mat &mask=Mat());

int affine_to_homogeneous( Mat &mat );

int homogeneous_to_affine( Mat &mat);

Point2f estimateTranslation(const vector<Point2f> &points1, const vector<Point2f> &points2);

Mat get_tsr_matrix( const vector<Point2f> &points1, const vector<Point2f> &points2 );

Mat get_tsrx_matrix( const vector<Point2f> &points1, const vector<Point2f> &points2, float lambda);

int get_scale_dxdy( const vector<Point2f> &points1, const vector<Point2f> &points2, Point2f &dxdy, float &scale);

Mat get_ts_matrix( const vector<Point2f> &points1, const vector<Point2f> &points2 );

double get_scale( const vector<Point2f> &points1, const vector<Point2f> &points2 );

int solveAffine(const vector<Point2f> &points1, const vector<Point2f> &points2, Mat affine, Mat preBgAffine, Mat &curBgAffine);

//int synthesis_new_img(const Mat &fg_img, Mat fg_alpha, const Mat &bg_img_warp, const Mat &bg_img, const int &dx, const int &dy,
//					  const int &fgFrameId, const int &bgVideoId, const int &bgFrameId, Mat &new_img, Mat &new_large_img);

int synthesis_new_img(const Mat &fg_img, Mat fg_alpha, Mat bg_img_warp, Mat bg_img, const int &dx, const int &dy,
					  const int &fgFrameId, const int &bgVideoId, const int &bgFrameId, Mat &new_img, Mat &new_large_img, int method, bool show_info=true);

int synthesis_new_img(const Mat &fg_img, Mat fg_alpha, Mat bg_img_warp, Mat &new_img);

int keyPointsToPoints(const vector<KeyPoint> &keyPoints, vector<Point2f> &points);

int synthesis_new_img(Mat foreground_img, Mat alpha, Mat background_img, int dx, int dy, Mat &new_frame);

void do_transform(const std::vector<cv::Point3f> &ipt, std::vector<cv::Point3f> &dpt, const cv::Mat &T);

void get_boundary(const std::vector<cv::Point3f> &ipt, int &xmin, int &xmax, int &ymin, int &ymax);

void showRegist(const cv::Mat &src, const cv::Mat &dest, const cv::Mat &T, int ntimes=-1, int delay=500);

void showFeatures(const cv::Mat &img, const std::vector<Point2f> &fea, int waitTime=0);

void showMatch(const cv::Mat &imi, const std::vector<Point2f> &pti, const cv::Mat &imj, const std::vector<Point2f> &ptj, const uchar *ptMask, int waitTime=0);


int refine_H(const cv::Mat &ref, const cv::Mat &src, cv::Mat &H, int nKLTFeatures=300, float ransacThreshold=3.0f, std::vector<Point2f> *refSel=NULL, std::vector<Point2f> *srcSel=NULL);

int refine_H(const cv::Mat &ref, const cv::Mat &src, const std::vector<Point2f> &srcKLT, cv::Mat &H,  float ransacThreshold=3.0f, std::vector<Point2f> *refSel=NULL, std::vector<Point2f> *srcSel=NULL);




