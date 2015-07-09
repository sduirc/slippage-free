
#pragma once

#include"dataset.h"
#include"imgset.h"

class CSlippage
{
public:
	std::string  dataDir;

	vector<_ISRPtrT> bgIsrList;
	_ISRPtrT		 fgIsrPtr;
	vector<_ISRPtrT> alphaIsrList;

	double			 m_bgWorkScale;
	double			 m_fgWorkScale;
	cv::Size		 m_bgWorkSize;

	int _LoadDataSet(FgDataSet *fg=NULL, BgDataSet *bg=NULL, BgDiskMap *bgm=NULL);

	int _SaveDataSet(FgDataSet *fg=NULL, BgDataSet *bg=NULL, BgDiskMap *bgm=NULL);
	
public:
	int Load(const std::string &dir, double bgProPixels=0.5);

	int SetFgWorkScale(double scale)
	{
		if(scale > 0)
		{
			m_fgWorkScale = scale;
			return 0;
		}
		else
		{
			return -1;
		}
	}

	cv::Size  GetFgSize() const
	{
		return cv::Size((int)(fgIsrPtr->Width() * m_fgWorkScale), (int)(fgIsrPtr->Height() * m_fgWorkScale));
	}


	cv::Point2f GetFgCenter() const
	{
		cv::Size size = GetFgSize();
		return cv::Point2f(size.width/2, size.height/2);
	//	return cv::Point2f(fgIsrPtr->Width()/2, fgIsrPtr->Height()/2);
	}

	cv::Size GetBgWorkSize() const
	{
	//	return cv::Size(bgIsrList.front()->Width(), bgIsrList.front()->Height());
		return m_bgWorkSize;
	}

	double  GetBgWorkScale() const
	{
		return m_bgWorkScale;
	}

	cv::Mat  GetInitTransform(double bgScale);

	const std::string &GetDataDir() const
	{
		return dataDir;
	}

	int UpdateFgData(const cv::Mat &featureMask, int footSmoothWSZ, int nKLTDetect=500, int nKLTSelect=100);

	int UpdateBgData(cv::Size KLTGrid=cv::Size(3,3), int nFeaInEachGrid=50);

	struct KNNParam
	{
	public:
		int		K0, K1; //K for the first and the second level
		int     D;      //Interval for the first level
		float   MAX_DST_ERR;
		float   MAX_TSF_ERR;
		float   MAX_REJECT_RATIO;
		int     INTERNEL_EXCLUDE_RANGE;
		int     INTERNEL_INCLUDE_RANGE;
		float   RANSAC_ERR;
		int     minPointsInEachGrid;
		cv::Size gridSize;
	public:
		KNNParam()
			:K0(2), K1(4), D(16), MAX_DST_ERR(0.05), MAX_REJECT_RATIO(0.75), MAX_TSF_ERR(16), INTERNEL_EXCLUDE_RANGE(50), INTERNEL_INCLUDE_RANGE(8), RANSAC_ERR(3), minPointsInEachGrid(16), gridSize(2,2)
		{
		}
	};

	int UpdateBgKNN(KNNParam param=KNNParam(), bool updateNN=true, bool findFarNbrs=true, bool updateTransform=true, const cv::Mat &feaMask=cv::Mat());

	int StitchBg();

	struct DPParam
	{
		int     method;
		int     motion[5];
		int		NM;
		float   WEIGHT_COVERAGE;
		float   WEIGHT_DISTORTION;
		float   WEIGHT_SMOOTH;
		float   WEIGHT_TRANSFORM;
		float   MAX_TSF_ERROR; //max transformation error to filter KNN
		float   MAX_DST_ERROR;
		int		foreground_extension[4]; // up, down, left, right
		int     firstFrameRange[2];
		int		foreground_start_end[2]; // default: start:0, end:-1
		bool  is_dynamic_bg;
	public:
		DPParam()
			:method(1), NM(2), WEIGHT_COVERAGE(200), WEIGHT_DISTORTION(10.0f), WEIGHT_SMOOTH(0.1), WEIGHT_TRANSFORM(100), MAX_TSF_ERROR(2), MAX_DST_ERROR(0.1), is_dynamic_bg(false)
		{
			motion[0]=MOTION_TS;
			motion[1]=MOTION_TSR;
			foreground_extension[0] = 0; //up
			foreground_extension[1] = 0; //down
			foreground_extension[2] = 0; //left
			foreground_extension[3] = 0; //right
			foreground_start_end[0] = 0; //start
			foreground_start_end[1] = -1;//end
		}
	};

	int UpdateCorrespondence(const cv::Mat &BGT, DPParam param=DPParam());

	int SolveBGTransform(const cv::Mat &BGT, DPParam param=DPParam());

	int GetShadowHomography(const std::vector<Point2f> &transformed_points, vector< vector<Mat> > &vH, DPParam param=DPParam());

	int Synthesis(bool stitch);

	int Output(DPParam param, const vector< vector<Mat>> &v_shadow_H = vector< vector<Mat> >(), float level=0.3, bool show_info=false );
	
	int OutputUnclip(DPParam param, const vector< vector<Mat> > &v_shadow_H = vector< vector<Mat> >(), float level = 0.3, bool show_foot_points = false );

	struct SSFParam
	{
		int width;
		int height;
		int interval_height;
		int selected_width;
		Scalar colors[5];
		Scalar selected_bar_color;
		Scalar selected_color;
		Scalar bg_color;

	public:
		SSFParam()
		{
			width = 960;
			height = 16;
			interval_height = 16;

			selected_width = 15;
			
			for(int i=0; i<5; i++)
			{
				colors[i] = Scalar(255, 128, 0);
			}
		
			selected_bar_color = Scalar(255, 128, 0);
			selected_color = Scalar(255, 255, 255);
			bg_color = Scalar(0, 0, 0);
		}
	};

	int ShowSelectedFrame(DPParam param, SSFParam ssf_param);

	struct SSFParam2
	{
		int height;
		Scalar start_color;
		Scalar end_color;

	public:
		SSFParam2()
		{
			height = 20;
			start_color = Scalar(0, 0, 255);
			end_color = Scalar(255, 0, 255);
		}
	};

	int ShowSelectedFrame2(DPParam param, SSFParam2 ssf_param);

};