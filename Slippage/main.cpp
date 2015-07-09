
#include "stdafx.h"

#define USE_LIBC_LIB
//#define USE_CAPX_LIB
#define USE_OPENCV2_LIB
#include "lib1.h"

#include"BFC\argv.h"
#include"BFC\fio.h"


#pragma comment(lib,_LN_OPENCV2(stitching))
#pragma comment(lib,_LN_OPENCV2(nonfree))


//#include "vfx\vfxserv.h"
//#pragma comment(lib,"vfxserv.lib")

#include"core.h"

cv::Mat MakeMask(const cv::Size &size, float start_row_ratio=0, float end_row_ratio=1.0f, float start_col_ratio=0.0f, float end_col_ratio=1.0f)
{
	cv::Mat mask = Mat::zeros(size.height, size.width, CV_8U);

	{
		int rows = size.height;
		int cols = size.width;
		int start_row = (int)(start_row_ratio * rows);
		int end_row = (int)(end_row_ratio * rows);
		int start_col = (int)(start_col_ratio * cols);
		int end_col = (int)(end_col_ratio * cols);
		Range row_range = Range(start_row, end_row);
		Range col_range = Range(start_col, end_col);
		Mat alpha_update_roi = mask(row_range, col_range);
		Mat alpha_temp = Mat::ones(end_row-start_row, end_col-start_col, CV_8U) * 255;
		alpha_temp.copyTo(alpha_update_roi);
	}

	return mask;
}

#include"SutherlandHodgman.h"

int main(int argc, char *argv[])
{
//	char* vdata[]={"","..\\data\\", "A1_girl","A2_penguin","A2_penguin2","A3_boy","A4_dance", "A5_goat", "A6_basketball", "A7_skater", "A8_rain", "A9_mj","A10_walk","A11_walk_seaside","A12_couple"};
//	argc=sizeof(vdata)/sizeof(vdata[0]);
//	argv=vdata;

	if(argc<=2)
	{
		printf("\nUsages : slippage.exe .\\data  A1_girl A2_penguin ...\n");
		system("pause");
		return 0;
	}

	CSlippage obj;

	std::string dir(argv[1]);
	dir+="\\";

	for(int n=2; n<argc; ++n)
	{
	
	obj.Load(dir+argv[n]);

	ff::HTBFile cfgFile;
	cfgFile.Load(obj.GetDataDir()+"\\config.txt");
	const std::string *paramStr=cfgFile.GetBlock("param");
	if(!paramStr)
		return -1;

	ff::CommandArgSet args, defaultArgs;
	args.SetArg(*paramStr);

	defaultArgs.SetArg(" -updateFg + -updateBg + -updateBgKNN +  -updateBgTransform + -footSoothWSZ 15 -dynamicBG - -useStitchBg -");
	args.SetNext(&defaultArgs);

	bool updateFg=args.Get<bool>("updateFg"), updateBg=args.Get<bool>("updateBg"), updateBgKNN=args.Get<bool>("updateBgKNN"), updateBgTransform=args.Get<bool>("updateBgTransform"), useStitchBg=args.Get<bool>("useStitchBg");

	obj.SetFgWorkScale( args.Get<float>("fgWorkScale") );

	float bgInitT[3];
	args.GetArray("bgInitTransform",bgInitT);

	cv::Mat BGT=obj.GetInitTransform(bgInitT[0]);	// scale of bg
	Mat BGT_T = (Mat_<double>(3,3) << 1, 0, bgInitT[1], 0, 1, bgInitT[2], 0, 0, 1);	// translation of bg
	BGT = BGT_T * BGT;
	cout<<BGT<<endl;

	SlippageData gdata;
	gdata.m_InitBT=BGT;
	gdata.m_bgWorkScale=obj.GetBgWorkScale();
	{
		ff::OBFStream os(ff::CatDirectory(obj.GetDataDir(),"global.db"));
		os<<gdata;
	}

	float maskROI[4];

	if(updateFg)
	{
		args.GetArray("fgMask",maskROI);
		cv::Mat fgFeatureMask( MakeMask(obj.GetFgSize(), maskROI[0],maskROI[1],maskROI[2],maskROI[3]) );
		obj.UpdateFgData(fgFeatureMask, args.Get<int>("footSmoothWSZ"));
	}

	if(updateBg)
	{
		obj.UpdateBgData();
	}

	CSlippage::KNNParam knnParam;
	if(updateBgKNN)
	{
		obj.UpdateBgKNN(knnParam,true,args.Get<bool>("useFarNbrs"),false);
	}

	if(updateBgTransform)
	{
		args.GetArray("bgMask",maskROI);
		cv::Mat bgFeatureMask( MakeMask(obj.GetBgWorkSize(),  maskROI[0],maskROI[1],maskROI[2],maskROI[3]) );

		obj.UpdateBgKNN(knnParam,false,false,true,bgFeatureMask);
	}

	if(useStitchBg)
		obj.StitchBg();
	
	CSlippage::DPParam dpParam;

	dpParam.MAX_DST_ERROR=args.Get<float>("MAX_DST_ERROR");		// default 0.1
	dpParam.MAX_TSF_ERROR=args.Get<float>("MAX_TSF_ERROR");		// default 2
	{
		std::vector<int> vmotion;
		args.GetVector("motion",vmotion);
		for(size_t i=0; i<vmotion.size(); ++i)
			dpParam.motion[i]=vmotion[i];
		dpParam.NM=(int)vmotion.size();
	}

	dpParam.method=args.Get<int>("dpMethod");
	dpParam.WEIGHT_COVERAGE=args.Get<float>("wc");	
	dpParam.WEIGHT_TRANSFORM=args.Get<float>("wt");
	dpParam.WEIGHT_DISTORTION=args.Get<float>("wd");
	dpParam.WEIGHT_SMOOTH=args.Get<float>("ws");	
	dpParam.is_dynamic_bg=args.Get<bool>("dynamicBG");

	args.GetArray("firstFrame",dpParam.firstFrameRange);
	args.GetArray("fgExtension",dpParam.foreground_extension);
	args.GetArray("fgStartEndPos",dpParam.foreground_start_end);

	obj.UpdateCorrespondence(BGT,dpParam);
	obj.SolveBGTransform(BGT, dpParam);

	vector< vector<Mat> > v_shadow_H;
	if(args.Get<bool>("shadow"))
	{
		std::vector<float> s;
		args.GetVector("shadowShape",s);
	
		vector<Point2f> v_transformed_corners;
		v_transformed_corners.push_back(Point2f(s[0], s[1]));
		v_transformed_corners.push_back(Point2f(s[2], s[3]));
		v_transformed_corners.push_back(Point2f(s[4], s[5]));
		v_transformed_corners.push_back(Point2f(s[6], s[7]));

		obj.GetShadowHomography(v_transformed_corners, v_shadow_H, dpParam);
	}

	vector<float> shadow_weight;
	args.GetVector<float>("shadow_weight", shadow_weight);
	obj.Output(dpParam, v_shadow_H, shadow_weight.size()>0 ? shadow_weight[0] : 0.3, false);

	}

	return 0;
}










#if 0

const double start_row_ratio = 0.4; //old bg
const double end_row_ratio = 1;
const double start_col_ratio = 0;
const double end_col_ratio = 1;

const double start_row_ratio2 = 0; //new bg
const double end_row_ratio2 = 1;
const double start_col_ratio2 = 0;
const double end_col_ratio2 = 1;

//int pos_x=150, pos_y=30;
int pos_x=0, pos_y=0;

const char DIR[20] = "D14";
char fgDir[50];
char alphaDir[50];
char bgDir[50];
char fgData[50];
char bgData[50];
char bgDataEx[50];
char result[50];
char result_large[50];
char result_v[50];

char fgDataTxt[50];
char bgDataTxt[50];
char framesDataTxt[50];






int main(int argc, char *argv[])
{
	sprintf( fgDir, "..\\data\\%s\\F\\", DIR );
	sprintf( alphaDir, "..\\data\\%s\\A\\", DIR );
	sprintf( bgDir, "..\\data\\%s\\B\\", DIR );
	sprintf( fgData, "..\\data\\%s\\fg.idx", DIR );
	sprintf( bgData, "..\\data\\%s\\bg.idx", DIR );
	sprintf( bgDataEx, "..\\data\\%s\\bg.dm", DIR );
	sprintf( result, "..\\data\\%s\\result.avi", DIR );
	sprintf( result_large, "..\\data\\%s\\result_large.avi", DIR );
	sprintf( result_v, "..\\data\\%s\\result_v.avi", DIR );
	sprintf( fgDataTxt, "..\\data\\%s\\fg.txt", DIR );
	sprintf( bgDataTxt, "..\\data\\%s\\bg.txt", DIR );
	sprintf( framesDataTxt, "..\\data\\%s\\frames.txt", DIR );

	FgDataSet fgDataSet;
	BgDataSet bgDataSet;
	BgDiskMap bgDiskMap;

	bgDiskMap.Load(bgDataEx);
	bgDataSet.Load(bgData);
	fgDataSet.Load(fgData);

	vector<_ISRPtrT> fgIsrList;
	vector<_ISRPtrT> alphaIsrList;
	vector<_ISRPtrT> bgIsrList;

	int ecb = LoadImageSet(_T(bgDir),bgIsrList);
	int ecf = LoadImageSet(_T(fgDir),fgIsrList);
	int eca = LoadImageSet(_T(alphaDir),alphaIsrList);

	ImageSetReader &fgIsr(*fgIsrList.front());
	ImageSetReader &alphaIsr(*alphaIsrList.front());
	assert( fgIsr.Size() == alphaIsr.Size() );

//	cv::Mat BGT=(cv::Mat_<double>(3,3)<<0.9,0,-200,0,0.9,0,0,0,1);
	cv::Mat BGT= (cv::Mat_<double>(3,3)<<1,0,0,0,1,0,0,0,1);
	cv::Mat BGTInv(BGT.inv());

	// bg roi
	int rows3 = (*bgIsrList.front()).Height();
	int cols3 = (*bgIsrList.front()).Width();
	int start_row3 = (int)(start_row_ratio2 * rows3);
	int end_row3 = (int)(end_row_ratio2 * rows3);
	int start_col3 = (int)(start_col_ratio2 * cols3);
	int end_col3 = (int)(end_col_ratio2 * cols3);
	Range row_range3 = Range(start_row3, end_row3);
	Range col_range3 = Range(start_col3, end_col3);
	Mat alpha_update3 = Mat::zeros( rows3, cols3, CV_8U );
	Mat alpha_update_roi3 = alpha_update3( row_range3, col_range3 );
	Mat alpha_temp3 = Mat::ones( end_row3 - start_row3, end_col3 - start_col3, CV_8U ) * 255;
	alpha_temp3.copyTo(alpha_update_roi3);





// 可视化特征
#if 0

	VideoWriter result_visual = VideoWriter(result_v, CV_FOURCC('D','I','V','X'), 25, Size(cols3, rows3));

	for(size_t i=0; i<bgDataSet.Size(); ++i)
	{
		std::cout << "bg:" << i << endl;

		BgFrameData *cur_fd = bgDataSet.GetAtIndex(i);
		int cur_video_id = cur_fd->m_videoID;
		int cur_frame_id = cur_fd->m_FrameID;

		bgIsrList[cur_video_id]->SetPos(cur_frame_id);
		Mat cur_img=bgIsrList[cur_video_id]->Read()->clone();

		char str[50];
		sprintf( str, "cur #:%d-%d", cur_video_id, cur_frame_id );
		putText( cur_img, str, Point(10, 35), FONT_HERSHEY_PLAIN, 3, Scalar(0, 0, 255), 3 );

		for(size_t j=0; j<cur_fd->m_nbr.size(); ++j)
		{
			result_visual.write(cur_img);
			cv::imshow("a", cur_img);
			cv::waitKey(1);

			BgFrameData *cpd_fd = bgDataSet.GetAtIndex(cur_fd->m_nbr[j].m_index);
			int cpd_video_id = cpd_fd->m_videoID;
			int cpd_frame_id = cpd_fd->m_FrameID;
			 
			bgIsrList[cpd_video_id]->SetPos(cpd_frame_id);
			Mat cpd_img=bgIsrList[cpd_video_id]->Read()->clone();

			cv::Mat T = cur_fd->m_nbr[j].m_T;
			affine_to_homogeneous(T);
			T = T.inv();
			if(T.rows==2)
				cv::warpAffine(cpd_img, cpd_img, T, cpd_img.size());
			else
				cv::warpPerspective(cpd_img,cpd_img,T,cpd_img.size());

			char str[50];
			sprintf( str, "cpd #:%d-%d", cpd_video_id, cpd_frame_id );
			putText( cpd_img, str, Point(10, 35), FONT_HERSHEY_PLAIN, 3, Scalar(0, 0, 255), 3 );

			result_visual.write(cpd_img);
			cv::imshow("a", cpd_img);
			cv::waitKey(1);
		}


	}

	result_visual.release();

#endif

	return 0;
}


#endif







