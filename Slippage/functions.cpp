#include "stdafx.h"
#include "functions.h"


using namespace cv;

int getKltkeyPoints( Mat img, vector<Point2f> &points, int nmax, Mat &mask )
{
	if( img.data != 0 && img.channels() != 1)
	{
		cvtColor(img, img ,CV_BGR2GRAY); 
	}
	if( mask.data != 0 && mask.channels() != 1)
	{
		cvtColor(mask, mask ,CV_BGR2GRAY); 
	}

	goodFeaturesToTrack( img, points, nmax, 0.01, 5.0, mask );  

	//cornerSubPix( img, points, cvSize(10, 10), cvSize(-1,-1), 
	//	cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03) );  
 
	return 0;
}

int KLTTrack( cv::Mat preImg, cv::Mat curImg, const vector<Point2f> &preKeyPoints, 
				   vector<Point2f> &preMatch, vector<Point2f> &curMatch)
{
	if( preImg.data != 0 && preImg.channels() != 1)
	{
		cvtColor(preImg, preImg ,CV_BGR2GRAY); 
	}
	if( curImg.data != 0 && curImg.channels() != 1)
	{
		cvtColor(curImg, curImg ,CV_BGR2GRAY); 
	}

	vector<uchar> featuresFound;
	vector<float> featuresError;

	vector<Point2f> keyPoints;
	calcOpticalFlowPyrLK( preImg, curImg,  preKeyPoints, keyPoints, featuresFound, featuresError );

	vector<Point2f> preMatchPoints, matchPoints;
	for(uint i=0; i<featuresFound.size(); i++)
	{
		if(featuresFound[i] == 1)
		{
			preMatchPoints.push_back(preKeyPoints[i]);
			matchPoints.push_back(keyPoints[i]);
		}
	}

	Mat ransacMask;
	Mat H = findHomography(preMatchPoints, matchPoints, CV_RANSAC, 3.0, ransacMask);

	vector<Point2f> preGoodMatchPoints;
	vector<Point2f> goodMatchPoints;
	uchar *p = ransacMask.ptr<uchar>();

	for (uint i=0; i<preMatchPoints.size(); i++)
	{
		if(p[i] != 0)
		{
			preGoodMatchPoints.push_back(preMatchPoints[i]);
			goodMatchPoints.push_back(matchPoints[i]);
		}
	}

//	similarity = estimateRigidTransform(preGoodMatchPoints, goodMatchPoints, false);

	preMatch.swap(preGoodMatchPoints);
	curMatch.swap(goodMatchPoints);

	return 0;
}


int KLTTrackNoRansac( cv::Mat preImg, cv::Mat curImg, const vector<Point2f> &preKeyPoints, 
				   vector<Point2f> &preMatch, vector<Point2f> &curMatch, int maxLevel)
{
	if( preImg.data != 0 && preImg.channels() != 1)
	{
		cvtColor(preImg, preImg ,CV_BGR2GRAY); 
	}
	if( curImg.data != 0 && curImg.channels() != 1)
	{
		cvtColor(curImg, curImg ,CV_BGR2GRAY); 
	}

	vector<uchar> featuresFound;
	vector<float> featuresError;

	vector<Point2f> keyPoints;
	calcOpticalFlowPyrLK( preImg, curImg,  preKeyPoints, keyPoints, featuresFound, featuresError, cv::Size(21,21), maxLevel);

	for(uint i=0; i<featuresFound.size(); i++)
	{
		if(featuresFound[i] == 1)
		{
			preMatch.push_back(preKeyPoints[i]);
			curMatch.push_back(keyPoints[i]);
		}
	}

	return 0;
}

cv::Mat findHomographyEx(const std::vector<Point2f> &prePoints, const std::vector<Point2f> &curPoints, int method, double ransacThreshold, std::vector<Point2f> *preGood, std::vector<Point2f> *curGood)
{
	Mat ransacMask;
	Mat H = findHomography(prePoints, curPoints, method, ransacThreshold, ransacMask);

	if(preGood || curGood)
	{
		vector<Point2f> preGoodMatchPoints;
		vector<Point2f> goodMatchPoints;
		uchar *p = ransacMask.ptr<uchar>();

		for (uint i=0; i<prePoints.size(); i++)
		{
			if(p[i] != 0)
			{
				preGoodMatchPoints.push_back(prePoints[i]);
				goodMatchPoints.push_back(curPoints[i]);
			}
		}

		if(preGood)
			preGood->swap(preGoodMatchPoints);

		if(curGood)
			curGood->swap(goodMatchPoints);
	}

	return H;
}

int getKltTranslation( cv::Mat preImg, cv::Mat curImg,  const vector<Point2f> &preKeyPoints, 
				   vector<Point2f> &keyPoints, cv::Mat &similarity, vector<Point2f> *preGood, vector<Point2f> *curGood)
{
	if( preImg.data != 0 && preImg.channels() != 1)
	{
		cvtColor(preImg, preImg ,CV_BGR2GRAY); 
	}
	if( curImg.data != 0 && curImg.channels() != 1)
	{
		cvtColor(curImg, curImg ,CV_BGR2GRAY); 
	}

	vector<uchar> featuresFound;
	vector<float> featuresError;

	calcOpticalFlowPyrLK( preImg, curImg,  preKeyPoints, keyPoints, featuresFound, featuresError, cv::Size(21,21), 6);

	vector<Point2f> preMatchPoints, matchPoints;
	for(uint i=0; i<featuresFound.size(); i++)
	{
		if(featuresFound[i] == 1)
		{
			preMatchPoints.push_back(preKeyPoints[i]);
			matchPoints.push_back(keyPoints[i]);
		}
	}

	similarity = findHomographyEx(preMatchPoints, matchPoints, CV_RANSAC, 4.0, preGood, curGood);

	return 0;
}


int getSiftKeyPoints( Mat img, vector<KeyPoint> &keyPoints, Mat &description, Mat &mask )
{
	if( mask.data != 0 && mask.channels() != 1)
	{
		cvtColor(mask, mask ,CV_BGR2GRAY); 
	}

	initModule_nonfree(); 
	
	Ptr<FeatureDetector> feature_detector = FeatureDetector::create( "SIFT" );  
	Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create( "SIFT" );  
	
	if (feature_detector.empty() || descriptor_extractor.empty())  
	{
		cout << "failed to create feature_detector!" << endl;
		return -1;
	}

	feature_detector->detect(img, keyPoints, mask);  
	descriptor_extractor->compute(img, keyPoints, description);   

	return 0;
}


int getSiftMatchCount(Mat descriptor1, Mat descriptor2, const vector<KeyPoint> &keyPoints1, 
				 const vector<KeyPoint> &keyPoints2, int &goodMatchCount, const Mat &mask)
{
	Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create( "BruteForce" ); 

	vector<DMatch> matches; 
	descriptor_matcher->match(descriptor1, descriptor2, matches); 
	
	vector<Point2f> points1, points2;
	for (uint k = 0; k < matches.size(); k++)
	{	
		Point2f pt1 = keyPoints1[ matches[k].queryIdx ].pt;
		Point2f pt2 = keyPoints2[ matches[k].trainIdx ].pt;

		if( mask.data == NULL || mask.at<uchar>(int(pt1.y),int(pt1.x)) != 0 && mask.at<uchar>(int(pt2.y),int(pt2.x)) != 0  )
		{
			points1.push_back(pt1);
			points2.push_back(pt2);
		}
	}

	// get the good matches using RANSAC
	if(points1.size() < 5)
	{
		goodMatchCount = points1.size();
	}
	else
	{
		Mat ransac_mask;
		Mat H = findHomography(points1, points2, CV_RANSAC, 16, ransac_mask);

		uchar *p = ransac_mask.ptr<uchar>();
		goodMatchCount = 0;

		for (uint k=0; k<points1.size(); k++)
		{
			if(p[k] != 0)
			{
				goodMatchCount ++;
			}
		}
	}
	
	return 0;
}





//float _get_distortion_error(const cv::Mat &T, const std::vector<cv::Point2f> &pt)
//{
//	float err=0;
//
//	std::vector<cv::Point2f>  dpt;
//	cv::perspectiveTransform(pt,dpt,T);
//
//	assert(dpt.size()==pt.size());
//
//	for(size_t i=0; i<pt.size(); ++i)
//	{
//		cv::Point2f dv(dpt[i]-pt[i]);
//		err+=dv.dot(dv);
//	}
//
//	err /= pt.size();
//
//	return err;
//}


float _get_transform_error(const cv::Mat &T, std::vector<cv::Point3f> &pt, const std::vector<cv::Point3f> &ref_pt)
{
	float err=0;

	std::vector<cv::Point3f>  dpt;
	cv::transform(pt,dpt,T);

	assert(dpt.size()==pt.size());

	for(size_t i=0; i<pt.size(); ++i)
	{
		dpt[i]*=1.0/dpt[i].z;
		cv::Point3f dv(dpt[i]-ref_pt[i]);

		err+=dv.dot(dv);
	}

	err /= pt.size();

	pt.swap(dpt);

	return err;
}


float _get_transform_error(const cv::Mat &T, std::vector<cv::Point2f> &pt, const std::vector<cv::Point2f> &ref_pt)
{
	float err=0;

	std::vector<cv::Point2f>  dpt;
	cv::perspectiveTransform(pt,dpt,T);

	assert(dpt.size()==pt.size());

	for(size_t i=0; i<pt.size(); ++i)
	{
		cv::Point2f dv(dpt[i]-ref_pt[i]);
		err+=dv.dot(dv);
	}

	err /= pt.size();

	pt.swap(dpt);

	return err;
}

float estimateShift(const std::vector<Point2f> &pti, const std::vector<Point2f> &ptj)
{
	float t=0;

	for(size_t i=0; i<pti.size(); ++i)
	{
		Point2f dv(ptj[i]-pti[i]);
		float L=dv.dot(dv);
		t+=sqrt(L);
	}

	t/=pti.size();

	return t;
}
//
//float getMatchError(const Mat &descriptor1, const Mat &descriptor2, const vector<Point2f> &keyPoints1, 
//					  const vector<Point2f> &keyPoints2, const std::vector<cv::Point3f> &pt, const Mat &mask)
//{
//	Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create( "BruteForce" ); 
//
//	vector<DMatch> matches; 
//	descriptor_matcher->match(descriptor1, descriptor2, matches); 
//	
//	vector<Point2f> points1, points2;
//	for (uint k = 0; k < matches.size(); k++)
//	{	
//		Point2f pt1 = keyPoints1[ matches[k].queryIdx ];
//		Point2f pt2 = keyPoints2[ matches[k].trainIdx ];
//
//		if( mask.data == NULL || mask.at<uchar>(int(pt1.y),int(pt1.x)) != 0 && mask.at<uchar>(int(pt2.y),int(pt2.x)) != 0  )
//		{
//			points1.push_back(pt1);
//			points2.push_back(pt2);
//		}
//	}
//
//	// get the good matches using RANSAC
//	if(points1.size() < 5)
//	{
//		return 1e6;
//	}
//	else
//	{
//		Mat ransac_mask;
//		Mat H = findHomography(points1, points2, CV_RANSAC, 16, ransac_mask);
//
//#if 0
//		uchar *p = ransac_mask.ptr<uchar>();
//		int goodMatchCount = 0;
//
//		for (uint k=0; k<points1.size(); k++)
//		{
//			if(p[k] != 0)
//			{
//				goodMatchCount ++;
//			}
//		}
//#endif
//
//		return _get_distortion_error(H,pt);
//	}
//	
//	return 0;
//}


int getSiftMatch(const Mat &descriptor1, const Mat &descriptor2, const vector<Point2f> &keyPoints1, 
				 const vector<Point2f> &keyPoints2, vector<Point2f> &goodPoints1, vector<Point2f> &goodPoints2, const Mat &mask)
{
	goodPoints1.clear();
	goodPoints2.clear();

	Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create( "BruteForce" ); 

	vector<DMatch> matches; 
	descriptor_matcher->match(descriptor1, descriptor2, matches); 
	
	vector<Point2f> points1, points2;
	for (uint k = 0; k < matches.size(); k++)
	{	
		Point2f pt1 = keyPoints1[ matches[k].queryIdx ];
		Point2f pt2 = keyPoints2[ matches[k].trainIdx ];

		if( mask.data == NULL || mask.at<uchar>(int(pt1.y),int(pt1.x)) != 0 && mask.at<uchar>(int(pt2.y),int(pt2.x)) != 0  )
		{
			points1.push_back(pt1);
			points2.push_back(pt2);
		}
	}

	// get the good matches using RANSAC

	if(points1.size() < 5)
	{
		goodPoints1 = points1;
		goodPoints2 = points2;
	}
	else
	{
		Mat ransac_mask;
		Mat H = findHomography(points1, points2, CV_RANSAC, 16, ransac_mask);

		uchar *p = ransac_mask.ptr<uchar>();
		for (unsigned int k=0; k<points1.size(); k++)
		{
			if(p[k] != 0)
			{
				goodPoints1.push_back(points1[k]);
				goodPoints2.push_back(points2[k]);
			}
		}
	}
	
	return 0;
}


int getSiftMatchNoRansac(const Mat &descriptor1, const Mat &descriptor2, const vector<Point2f> &keyPoints1, 
				 const vector<Point2f> &keyPoints2, vector<Point2f> &goodPoints1, vector<Point2f> &goodPoints2, const Mat &mask)
{
	goodPoints1.clear();
	goodPoints2.clear();

	Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create( "BruteForce" ); 

	vector<DMatch> matches; 
	descriptor_matcher->match(descriptor1, descriptor2, matches); 
	
	for (uint k = 0; k < matches.size(); k++)
	{	
		Point2f pt1 = keyPoints1[ matches[k].queryIdx ];
		Point2f pt2 = keyPoints2[ matches[k].trainIdx ];

		if( mask.data == NULL || mask.at<uchar>(int(pt1.y),int(pt1.x)) != 0 && mask.at<uchar>(int(pt2.y),int(pt2.x)) != 0  )
		{
			goodPoints1.push_back(pt1);
			goodPoints2.push_back(pt2);
		}
	}

	
	return 0;
}


int getSiftMatch(const Mat &descriptor1, const Mat &descriptor2, const vector<Point2f> &keyPoints1, const vector<Point2f> &keyPoints2, 
				 vector<Point2f> &goodPoints1, vector<Point2f> &goodPoints2, vector<BgFrameData::MyMatch> &myMatches, const Mat &mask)
{
	goodPoints1.clear();
	goodPoints2.clear();

	Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create( "BruteForce" ); 

	vector<DMatch> matches; 
	descriptor_matcher->match(descriptor1, descriptor2, matches); 
	
	vector<Point2f> points1, points2;
	for (uint k = 0; k < matches.size(); k++)
	{	
		Point2f pt1 = keyPoints1[ matches[k].queryIdx ];
		Point2f pt2 = keyPoints2[ matches[k].trainIdx ];

		if( mask.data == NULL || mask.at<uchar>(int(pt1.y),int(pt1.x)) != 0 && mask.at<uchar>(int(pt2.y),int(pt2.x)) != 0  )
		{
			points1.push_back(pt1);
			points2.push_back(pt2);
		}

		myMatches.push_back(BgFrameData::MyMatch((ushort)(matches[k].queryIdx), (ushort)(matches[k].trainIdx)));
	}

	// get the good matches using RANSAC

	if(points1.size() < 5)
	{
		goodPoints1 = points1;
		goodPoints2 = points2;
	}
	else
	{
		Mat ransac_mask;
		Mat H = findHomography(points1, points2, CV_RANSAC, 16, ransac_mask);

		uchar *p = ransac_mask.ptr<uchar>();
		for (unsigned int k=0; k<points1.size(); k++)
		{
			if(p[k] != 0)
			{
				goodPoints1.push_back(points1[k]);
				goodPoints2.push_back(points2[k]);
			}
		}
	}
	
	return 0;
}


int getSiftMatchTopN(const Mat &descriptor1, const Mat &descriptor2, const vector<KeyPoint> &keyPoints1, 
				 const vector<KeyPoint> &keyPoints2, vector<Point2f> &goodPoints1, vector<Point2f> &goodPoints2, const int &n, const Mat &mask)
{
	goodPoints1.clear();
	goodPoints2.clear();

	Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create( "BruteForce" ); 

	vector<DMatch> matches; 
	descriptor_matcher->match(descriptor1, descriptor2, matches);

	sort(matches.begin(), matches.end());
	int use_match_count = min(n, (int)(matches.size()));

	vector<Point2f> points1, points2;
	for (uint k = 0; k < matches.size(); k++)
	{	
		Point2f pt1 = keyPoints1[ matches[k].queryIdx ].pt;
		Point2f pt2 = keyPoints2[ matches[k].trainIdx ].pt;

		if( mask.data == NULL || mask.at<uchar>(int(pt1.y),int(pt1.x)) != 0 && mask.at<uchar>(int(pt2.y),int(pt2.x)) != 0  )
		{
			points1.push_back(pt1);
			points2.push_back(pt2);
		}
	}

	// get the good matches using RANSAC
	if(points1.size() < 5)
	{
		goodPoints1 = points1;
		goodPoints2 = points2;
	}
	else
	{
		Mat ransac_mask;
		Mat H = findHomography(points1, points2, CV_RANSAC, 16, ransac_mask);
		int count = 0;

		uchar *p = ransac_mask.ptr<uchar>();
		for (unsigned int k=0; k<points1.size(); k++)
		{
			if(p[k] != 0 && count < use_match_count)
			{
				goodPoints1.push_back(points1[k]);
				goodPoints2.push_back(points2[k]);
				count ++;
			}
		}
	}
	
	return 0;
}


int affine_to_homogeneous( Mat &mat )
{
	if(mat.rows==2)
	{
		double a11 = mat.at<double>(0, 0);
		double a12 = mat.at<double>(0, 1);
		double a13 = mat.at<double>(0, 2);
		double a21 = mat.at<double>(1, 0);
		double a22 = mat.at<double>(1, 1);
		double a23 = mat.at<double>(1, 2);

		mat = (Mat_<double>(3, 3) << a11, a12, a13, a21, a22, a23, 0, 0, 1);
	}

	return 0;
}


int homogeneous_to_affine( Mat &mat)
{
	if(mat.rows==3)
	{
		double a11 = mat.at<double>(0, 0);
		double a12 = mat.at<double>(0, 1);
		double a13 = mat.at<double>(0, 2);
		double a21 = mat.at<double>(1, 0);
		double a22 = mat.at<double>(1, 1);
		double a23 = mat.at<double>(1, 2);

		mat = (Mat_<double>(2, 3) << a11, a12, a13, a21, a22, a23);
	}
	return 0;
}


Point2f estimateTranslation(const vector<Point2f> &points1, const vector<Point2f> &points2)
{
	Point2f translation;

	for( uint i=0; i<points1.size(); i++ )
	{
		translation += points2[i] - points1[i];
	}
	if ( points1.size() != 0 )
	{
		translation.x /= points1.size();
		translation.y /= points1.size();
	}

	return translation;
}


Mat get_tsr_matrix( const vector<Point2f> &points1, const vector<Point2f> &points2 )
{
	// 转换成解方程 A * X = B
	// X1 * S * cos - Y1 * sin + dx = X2;
	// X1 * sin + Y1 * S * cos + dy = Y2;

	if(points1.size() < 2)
	{
		cout << "get_tsr_matrix():not enough points." << endl;
		return Mat();
	}

	int rows = 2 * points1.size();
	Mat A = Mat::zeros( rows, 4, CV_64F );
	Mat X = Mat::zeros( 4, 1, CV_64F );
	Mat B = Mat::zeros( rows, 1, CV_64F );

	for( unsigned int i=0; i<points1.size(); i++)
	{
		A.at<double>(2*i, 0) = points1[i].x;
		A.at<double>(2*i, 1) = -points1[i].y;
		A.at<double>(2*i, 2) = 1;
		A.at<double>(2*i, 3) = 0;

		A.at<double>(2*i+1, 0) = points1[i].y;
		A.at<double>(2*i+1, 1) = points1[i].x;
		A.at<double>(2*i+1, 2) = 0;
		A.at<double>(2*i+1, 3) = 1;

		B.at<double>(2*i, 0) = points2[i].x;
		B.at<double>(2*i+1, 0) = points2[i].y;
	}

	solve( A, B, X, DECOMP_SVD ); 

	double s_cos = X.at<double>(0, 0);
	double sin = X.at<double>(1, 0);
	double dx = X.at<double>(2, 0);
	double dy = X.at<double>(3, 0);
	Mat tsr = (Mat_<double>(2, 3) << s_cos, -sin, dx, sin, s_cos, dy);

	return tsr;
}


Mat get_tsrx_matrix( const vector<Point2f> &points1, const vector<Point2f> &points2, float lambda)
{
	// 转换成解方程 A * X = B
	// X1 * S * cos - Y1 * sin + dx = X2;
	// X1 * sin + Y1 * S * cos + dy = Y2;

	if(points1.size() < 2)
	{
		cout << "get_tsr_matrix():not enough points." << endl;
		return Mat();
	}

	int rows = 2 * points1.size()+1;
	Mat A = Mat::zeros( rows, 4, CV_64F );
	Mat X = Mat::zeros( 4, 1, CV_64F );
	Mat B = Mat::zeros( rows, 1, CV_64F );

	for( unsigned int i=0; i<points1.size(); i++)
	{
		A.at<double>(2*i, 0) = points1[i].x;
		A.at<double>(2*i, 1) = -points1[i].y;
		A.at<double>(2*i, 2) = 1;
		A.at<double>(2*i, 3) = 0;

		A.at<double>(2*i+1, 0) = points1[i].y;
		A.at<double>(2*i+1, 1) = points1[i].x;
		A.at<double>(2*i+1, 2) = 0;
		A.at<double>(2*i+1, 3) = 1;

		B.at<double>(2*i, 0) = points2[i].x;
		B.at<double>(2*i+1, 0) = points2[i].y;
	}

	A.at<double>(rows-1,0)=0;
	A.at<double>(rows-1,1)=lambda;
	A.at<double>(rows-1,2)=0;
	A.at<double>(rows-1,3)=0;
	B.at<double>(rows-1,0)=0;

	solve( A, B, X, DECOMP_SVD ); 

	double s_cos = X.at<double>(0, 0);
	double sin = X.at<double>(1, 0);
	double dx = X.at<double>(2, 0);
	double dy = X.at<double>(3, 0);
	Mat tsr = (Mat_<double>(2, 3) << s_cos, -sin, dx, sin, s_cos, dy);

	return tsr;
}

Mat get_ts_matrix( const vector<Point2f> &points1, const vector<Point2f> &points2 )
{
	// 转换成解方程 A * X = B
	// X1 * S  + dx = X2;
	// Y1 * S  + dy = Y2;

	if(points1.size() < 2)
	{
		cout << "get_tsr_matrix():not enough points." << endl;
		return Mat();
	}

	int rows = 2 * points1.size();
	Mat A = Mat::zeros( rows, 3, CV_64F );
	Mat X = Mat::zeros( 3, 1, CV_64F );
	Mat B = Mat::zeros( rows, 1, CV_64F );

	for( unsigned int i=0; i<points1.size(); i++)
	{
		A.at<double>(2*i, 0) = points1[i].x;
		A.at<double>(2*i, 1) = 1;
		A.at<double>(2*i, 2) = 0;

		A.at<double>(2*i+1, 0) = points1[i].y;
		A.at<double>(2*i+1, 1) = 0;
		A.at<double>(2*i+1, 2) = 1;

		B.at<double>(2*i, 0) = points2[i].x;
		B.at<double>(2*i+1, 0) = points2[i].y;
	}

	solve( A, B, X, DECOMP_SVD ); 

	double scale = X.at<double>(0, 0);
	double dx = X.at<double>(1, 0);
	double dy = X.at<double>(2, 0);
	Mat tsr = (Mat_<double>(2, 3) << scale, 0, dx, 0, scale, dy);

	return tsr;
}


int get_scale_dxdy( const vector<Point2f> &points1, const vector<Point2f> &points2, Point2f &dxdy, float &scale )
{
	if(points1.size() < 2)
	{
		cout << "get_scale_dxdy():not enough points." << endl;
		return -1;
	}

	Mat tsr = get_tsr_matrix( points1, points2 );

	double s_cos = tsr.at<double>(0, 0);
	double s_sin = tsr.at<double>(1, 0);
//	scale = (float)(s_cos/sqrt(1-sin*sin));
	scale=sqrt(s_cos*s_cos+s_sin*s_sin);

#if 0
	for (uint k=0; k<points1.size(); k++)
	{
		dxdy += (points2[k] - points1[k]);
	}
	
	if(points1.size() != 0)
	{
		dxdy = Point2f(dxdy.x/points1.size(), dxdy.y/points1.size());
	}
#else
	dxdy=Point2f(tsr.at<double>(0,2),tsr.at<double>(1,2));
#endif

	return 0;
}


double get_scale( const vector<Point2f> &points1, const vector<Point2f> &points2 )
{
	if(points1.size() < 2)
	{
		cout << "get_scale_dxdy():not enough points." << endl;
		return -1;
	}

	Mat tsr = get_tsr_matrix( points1, points2 );

	double s_cos = tsr.at<double>(0, 0);
	double s_sin = tsr.at<double>(1, 0);
	double scale = sqrt(s_cos*s_cos + s_sin*s_sin);
//	float scale = (float)(s_cos/sqrt(1-sin*sin));

	return scale;
}


int solveAffine(const vector<Point2f> &points1, const vector<Point2f> &points2, Mat affine, Mat preBgAffine, Mat &curBgAffine)
{
	if(points1.size() < 2)
	{
		cout << "Error::solveAffine():not enough points." << endl;
		return -1;
	}

	// 解方程 T * A1 * P1 = A2 * P2, 现在已知A1，递推A2， 等式左边为已知, 求A2
	vector<Point2f> T_A1_P1;
	Mat T = affine;
	Mat A1 = preBgAffine;

	affine_to_homogeneous( A1 );

	Mat T_A1 = T * A1;
	homogeneous_to_affine( T_A1 );

	transform( points1, T_A1_P1, T_A1 );

	// 现在方程为 : T_A1_P1 = A2 * P2, 求A2
	curBgAffine = get_tsr_matrix( points2, T_A1_P1);

	return 0;
}


int synthesis_new_img(const Mat &fg_img, Mat fg_alpha, const Mat &bg_img_warp_, const Mat &bg_img, const int &dx, const int &dy,
					  const int &fgFrameId, const int &bgVideoId, const int &bgFrameId, Mat &new_img, Mat &new_large_img)
{
	Mat bg_img_warp(bg_img_warp_);

	CV_Assert(fg_img.depth() != sizeof(uchar));
	CV_Assert(bg_img_warp.depth() != sizeof(uchar));
	CV_Assert(fg_alpha.depth() != sizeof(uchar));

	if( fg_img.channels() != 3 || bg_img_warp.channels() != 3 )
	{
		return -1;
	}

	if( fg_alpha.channels() != 1 )
	{
		cvtColor(fg_alpha, fg_alpha ,CV_BGR2GRAY); 
	}

	int rows = fg_img.rows;
	int cols = fg_img.cols;
	int rows2 = bg_img_warp.rows;
	int cols2 = bg_img_warp.cols;

	
	
//	Mat affine_mat = (Mat_<float>(2,3) << 1, 0, cvRound(-dx), 0, 1, cvRound(-dy));
//	warpAffine(bg_img_warp, bg_img_warp, affine_mat, Size(fg_img.cols, fg_img.rows));


//	new_img = Mat::zeros( rows2, cols2, fg_img.type() );
	new_img=bg_img_warp.clone();

	for( int i = 0; i < rows2; ++i )
	{
		int fy=i-dy;

		if(uint(fy)<fg_img.rows)
		{
			const uchar* p_foreground = fg_img.ptr<uchar>(fy);
			const uchar* p_background = bg_img_warp.ptr<uchar>(i);
			const uchar* p_alpha = fg_alpha.ptr<uchar>(fy);
			uchar* p_new_frame = new_img.ptr<uchar>(i);

			for( int j = 0; j < cols2; ++j )
			{
				int fx=j-dx;

				if(uint(fx)<fg_img.cols)
				{
					p_new_frame[3 * j + 0] = (p_foreground[3 * fx + 0] * p_alpha[j] + p_background[3 * j + 0] * (255 - p_alpha[fx])) / 255;
					p_new_frame[3 * j + 1] = (p_foreground[3 * fx + 1] * p_alpha[j] + p_background[3 * j + 1] * (255 - p_alpha[fx])) / 255;
					p_new_frame[3 * j + 2] = (p_foreground[3 * fx + 2] * p_alpha[j] + p_background[3 * j + 2] * (255 - p_alpha[fx])) / 255;
				}
			}
		}
	}

	new_img=new_img(cv::Rect(dx,dy,cols,rows));
	cv::Mat bg_imgx=bg_img(cv::Rect(dx,dy,cols,rows));

//	imshow("new",new_img);
//	cv::waitKey(0);

	// print frame info on image.
	char str[50];
	sprintf( str, "#:%d--#:%d-%d", fgFrameId, bgVideoId, bgFrameId );
	putText( new_img, str, Point(10, 35), FONT_HERSHEY_PLAIN, 3, Scalar(0, 0, 255), 3 );

	// integrate multi video
	new_large_img = Mat::zeros( rows, 3*cols+20, fg_img.type() );

	Mat old_img_roi = new_large_img(Range(0, rows), Range(0, cols));
	Mat new_img_roi = new_large_img(Range(0, rows), Range(cols+10, 2*cols+10));
	Mat bg_img_roi = new_large_img(Range(0, rows), Range(2*cols+20, 3*cols+20));

	fg_img.copyTo(old_img_roi);
	new_img.copyTo(new_img_roi);
	bg_imgx.copyTo(bg_img_roi);
	
	return 0;
}

int synthesis_new_img(const Mat &fg_img, Mat fg_alpha, Mat bg_img_warp, Mat bg_img, const int &dx, const int &dy,
					  const int &fgFrameId, const int &bgVideoId, const int &bgFrameId, Mat &new_img, Mat &new_large_img, int method, bool show_info)
{
	CV_Assert(fg_img.depth() != sizeof(uchar));
	CV_Assert(bg_img_warp.depth() != sizeof(uchar));
	CV_Assert(fg_alpha.depth() != sizeof(uchar));

	if( fg_img.channels() != 3 || bg_img_warp.channels() != 3 )
	{
		return -1;
	}
	if( fg_alpha.channels() != 1 )
	{
		cvtColor(fg_alpha, fg_alpha ,CV_BGR2GRAY); 
	}

	int rows = fg_img.rows;
	int cols = fg_img.cols;
	int rows2 = bg_img_warp.rows;
	int cols2 = bg_img_warp.cols;

	if( dx < 0 )
	{
		copyMakeBorder( bg_img_warp, bg_img_warp, 0, 0, -dx, 0, BORDER_CONSTANT);
	}
	if( dy < 0 )
	{
		copyMakeBorder( bg_img_warp, bg_img_warp, -dy, 0, 0, 0, BORDER_CONSTANT);
	}
	if( dx > 0 )
	{
		Mat affine_mat = (Mat_<float>(2,3) << 1, 0, cvRound(-dx), 0, 1, 0);
		warpAffine(bg_img_warp, bg_img_warp, affine_mat, Size(bg_img_warp.cols, bg_img_warp.rows));
	}
	if( dy > 0 )
	{
		Mat affine_mat = (Mat_<float>(2,3) << 1, 0, 0, 0, 1, cvRound(-dy));
		warpAffine(bg_img_warp, bg_img_warp, affine_mat, Size(bg_img_warp.cols, bg_img_warp.rows));
	}
	if( cols > cols2 )
	{
		copyMakeBorder( bg_img_warp, bg_img_warp, 0, 0, 0, cols-cols2, BORDER_CONSTANT);
	}
	if( rows > rows2 )
	{
		copyMakeBorder( bg_img_warp, bg_img_warp, 0, rows-rows2, 0, 0, BORDER_CONSTANT);
	}

	new_img = Mat::zeros( rows, cols, fg_img.type() );
	new_large_img = Mat::zeros( rows, 3*cols+20, fg_img.type() );

	for( int i = 0; i < rows; ++i )
	{
		const uchar* p_foreground = fg_img.ptr<uchar>(i);
		const uchar* p_background = bg_img_warp.ptr<uchar>(i);
		const uchar* p_alpha = fg_alpha.ptr<uchar>(i);
		uchar* p_new_frame = new_img.ptr<uchar>(i);

		for( int j = 0; j < cols; ++j )
		{
			p_new_frame[3 * j + 0] = (p_foreground[3 * j + 0] * p_alpha[j] + p_background[3 * j + 0] * (255 - p_alpha[j])) / 255;
			p_new_frame[3 * j + 1] = (p_foreground[3 * j + 1] * p_alpha[j] + p_background[3 * j + 1] * (255 - p_alpha[j])) / 255;
			p_new_frame[3 * j + 2] = (p_foreground[3 * j + 2] * p_alpha[j] + p_background[3 * j + 2] * (255 - p_alpha[j])) / 255;
		}
	}

	Mat old_img_roi = new_large_img(Range(0, rows), Range(0, cols));
	Mat new_img_roi = new_large_img(Range(0, rows), Range(cols+10, 2*cols+10));
	Mat bg_img_roi = new_large_img(Range(0, rows), Range(2*cols+20, 3*cols+20));

	resize(bg_img, bg_img, Size(cols, rows));
	fg_img.copyTo(old_img_roi);
	new_img.copyTo(new_img_roi);
	bg_img.copyTo(bg_img_roi);

	if(show_info)
	{
		char str[50];
		sprintf( str, "#:%d--#:%d-%d : M%d", fgFrameId, bgVideoId, bgFrameId, method );
		putText( new_large_img, str, Point(20+cols, 35), FONT_HERSHEY_PLAIN, 3, Scalar(0, 0, 255), 3 );
	}
	
	return 0;
}


int synthesis_new_img(const Mat &fg_img, Mat fg_alpha, Mat bg_img_warp, Mat &new_img)
{
	CV_Assert(fg_img.depth() != sizeof(uchar));
	CV_Assert(bg_img_warp.depth() != sizeof(uchar));
	CV_Assert(fg_alpha.depth() != sizeof(uchar));

	if( fg_img.channels() != 3 || bg_img_warp.channels() != 3 )
	{
		return -1;
	}
	if( fg_alpha.channels() != 1 )
	{
		cvtColor(fg_alpha, fg_alpha ,CV_BGR2GRAY); 
	}

	int rows = fg_img.rows;
	int cols = fg_img.cols;
	int rows2 = bg_img_warp.rows;
	int cols2 = bg_img_warp.cols;

	if( cols > cols2 )
	{
		copyMakeBorder( bg_img_warp, bg_img_warp, 0, 0, 0, cols-cols2, BORDER_CONSTANT);
	}
	if( rows > rows2 )
	{
		copyMakeBorder( bg_img_warp, bg_img_warp, 0, rows-rows2, 0, 0, BORDER_CONSTANT);
	}

	new_img = Mat::zeros( rows, cols, fg_img.type() );

	for( int i = 0; i < rows; ++i )
	{
		const uchar* p_foreground = fg_img.ptr<uchar>(i);
		const uchar* p_background = bg_img_warp.ptr<uchar>(i);
		const uchar* p_alpha = fg_alpha.ptr<uchar>(i);
		uchar* p_new_frame = new_img.ptr<uchar>(i);

		for( int j = 0; j < cols; ++j )
		{
			p_new_frame[3 * j + 0] = (p_foreground[3 * j + 0] * p_alpha[j] + p_background[3 * j + 0] * (255 - p_alpha[j])) / 255;
			p_new_frame[3 * j + 1] = (p_foreground[3 * j + 1] * p_alpha[j] + p_background[3 * j + 1] * (255 - p_alpha[j])) / 255;
			p_new_frame[3 * j + 2] = (p_foreground[3 * j + 2] * p_alpha[j] + p_background[3 * j + 2] * (255 - p_alpha[j])) / 255;
		}
	}
	
	return 0;
}


int keyPointsToPoints(const vector<KeyPoint> &keyPoints, vector<Point2f> &points)
{
	points.clear();
	for(uint i=0; i<keyPoints.size(); i++)
	{
		points.push_back(keyPoints[i].pt);
	}
	return 0;
}


int synthesis_new_img(Mat foreground_img, Mat alpha, Mat background_img, int dx, int dy, Mat &new_frame)
{
	CV_Assert(foreground_img.depth() != sizeof(uchar));
	CV_Assert(background_img.depth() != sizeof(uchar));
	CV_Assert(alpha.depth() != sizeof(uchar));

	if( foreground_img.channels() != 3 || background_img.channels() != 3 )
	{
		return -1;
	}
	if( alpha.channels() != 1 )
	{
		cvtColor(alpha, alpha ,CV_BGR2GRAY); 
	}

	int rows = foreground_img.rows;
	int cols = foreground_img.cols;
	int rows2 = background_img.rows;
	int cols2 = background_img.cols;
	int new_rows = max( rows2, rows + dy );
	int new_cols = max( cols2, cols + dx );
	if( dx < 0 )
	{
		new_cols = new_cols - dx;
	}
	if( dy < 0 )
	{
		new_rows = new_rows - dy;
	}
	new_frame = Mat::zeros( new_rows, new_cols, foreground_img.type() );

	if( dx >= 0 && dy >= 0 )
	{
		copyMakeBorder( foreground_img, foreground_img, dy, (new_rows - dy - rows), dx, (new_cols - dx - cols), BORDER_CONSTANT);
		copyMakeBorder( alpha, alpha, dy, (new_rows - dy - rows), dx, (new_cols - dx - cols), BORDER_CONSTANT);
		copyMakeBorder( background_img, background_img, 0, (new_rows - rows2), 0, (new_cols - cols2), BORDER_CONSTANT);
	}
	else if( dx >= 0 && dy < 0 )
	{
		copyMakeBorder( foreground_img, foreground_img, 0, (new_rows - rows), dx, (new_cols - dx - cols), BORDER_CONSTANT);
		copyMakeBorder( alpha, alpha, 0, (new_rows - rows), dx, (new_cols - dx - cols), BORDER_CONSTANT);
		copyMakeBorder( background_img, background_img, -dy, (new_rows + dy - rows2), 0, (new_cols - cols2), BORDER_CONSTANT);
	}
	else if( dx < 0 && dy >= 0 )
	{
		copyMakeBorder( foreground_img, foreground_img, dy, (new_rows - dy - rows), 0, (new_cols - cols), BORDER_CONSTANT);
		copyMakeBorder( alpha, alpha, dy, (new_rows - dy - rows), 0, (new_cols - cols), BORDER_CONSTANT);
		copyMakeBorder( background_img, background_img, 0, (new_rows - rows2), -dx, (new_cols + dx - cols2), BORDER_CONSTANT);
	}
	else if( dx < 0 && dy < 0 )
	{
		copyMakeBorder( foreground_img, foreground_img, 0, (new_rows - rows), 0, (new_cols - cols), BORDER_CONSTANT);
		copyMakeBorder( alpha, alpha, 0, (new_rows - rows), 0, (new_cols - cols), BORDER_CONSTANT);
		copyMakeBorder( background_img, background_img, -dy, (new_rows + dy - rows2), -dx, (new_cols + dx - cols2), BORDER_CONSTANT);
	}

	for( int i = 0; i < new_rows; ++i )
	{
		uchar* p_foreground = foreground_img.ptr<uchar>(i);
		uchar* p_background = background_img.ptr<uchar>(i);
		uchar* p_alpha = alpha.ptr<uchar>(i);
		uchar* p_new_frame = new_frame.ptr<uchar>(i);

		for( int j = 0; j < new_cols; ++j )
		{
			p_new_frame[3 * j + 0] = (p_foreground[3 * j + 0] * p_alpha[j] + p_background[3 * j + 0] * (255 - p_alpha[j])) / 255;
			p_new_frame[3 * j + 1] = (p_foreground[3 * j + 1] * p_alpha[j] + p_background[3 * j + 1] * (255 - p_alpha[j])) / 255;
			p_new_frame[3 * j + 2] = (p_foreground[3 * j + 2] * p_alpha[j] + p_background[3 * j + 2] * (255 - p_alpha[j])) / 255;
		}
	}

	return 0;
}



void do_transform(const std::vector<cv::Point3f> &ipt, std::vector<cv::Point3f> &dpt, const cv::Mat &T)
{
	cv::transform(ipt,dpt,T);

	for(size_t i=0; i<dpt.size(); ++i)
	{
		dpt[i]*=1.0/dpt[i].z;
	}
}


void get_boundary(const std::vector<cv::Point3f> &ipt, int &xmin, int &xmax, int &ymin, int &ymax)
{
	xmin = INT_MAX;
	ymin = INT_MAX;
	xmax = INT_MIN;
	ymax = INT_MIN;

	for(size_t i=0; i<ipt.size(); ++i)
	{
		if(ipt[i].x < xmin)
			xmin = ipt[i].x;
		if(ipt[i].x > xmax)
			xmax = ipt[i].x;
		if(ipt[i].y < ymin)
			ymin = ipt[i].y;
		if(ipt[i].y > ymax)
			ymax = ipt[i].y;
	}
}

void showRegist(const cv::Mat &src, const cv::Mat &dest, const cv::Mat &T, int ntimes, int delay)
{
	cv::Mat warp;
	if(T.rows==3)
		cv::warpPerspective(src,warp,T,src.size());
	else
		cv::warpAffine(src,warp,T,src.size());

	for(int i=0; i<uint(ntimes); ++i)
	{
		cv::imshow("reg",i&1? dest : warp);
		cv::waitKey(delay);
	}
}

void pt2kp(const std::vector<Point2f> &pt, std::vector<cv::KeyPoint> &kp)
{
	kp.resize(pt.size());

	for(size_t i=0; i<pt.size(); ++i)
	{
		kp[i].pt=pt[i];
	}
}


void showFeatures(const cv::Mat &img, const std::vector<Point2f> &fea, int waitTime)
{
	cv::Mat dimg;

	std::vector<cv::KeyPoint> kp;
	pt2kp(fea,kp);
	cv::drawKeypoints(img,kp,dimg);

	cv::imshow("feature", dimg);
	cv::waitKey(waitTime);
}

void showMatch(const cv::Mat &imi, const std::vector<Point2f> &pti, const cv::Mat &imj, const std::vector<Point2f> &ptj, const uchar *ptMask, int waitTime)
{
	vector<Point2f> ransacPoints1, ransacPoints2;
	if(ptMask)
	{
		for (unsigned int k=0; k<pti.size(); k++)
		{
			if(ptMask[k] != 0)
			{
				ransacPoints1.push_back(pti[k]);
				ransacPoints2.push_back(ptj[k]);
			}
		}
	}
	else
	{
		ransacPoints1=pti;
		ransacPoints2=ptj;
	}

	const cv::Mat *preImg=&imi;
	const cv::Mat *curImg=&imj;
	cv::Mat  match;

	cv::KeyPoint kpi;
	std::vector<cv::KeyPoint> kp1,kp2;
	std::vector<cv::DMatch>  kpMatch;

	for(size_t i=0; i<ransacPoints1.size(); ++i)
	{
		kpi.pt=ransacPoints1[i];
		kp1.push_back(kpi);

		kpi.pt=ransacPoints2[i];
		kp2.push_back(kpi);
		kpMatch.push_back(DMatch(i,i,0));
	}

	cv::drawMatches(*preImg,kp1,*curImg,kp2,kpMatch,match);

	cv::imwrite("match.jpg",match);
	cv::imshow("match",match);
	cv::waitKey(waitTime); 
}

