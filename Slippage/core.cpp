
#include"stdafx.h"

#include"core.h"

#include"BFC\stdf.h"
#include"ZKIT\filex.h"
#include"functions.h"
#include"util.h"

#include"DP.h"

#include<functional>

#include"IPF\ipf.h"

int LoadImageSet(const string_t &path, std::vector<_ISRPtrT> &isList)
{
	int ec=-1;
	std::vector<string_t>  vName;
	if(ListSubDirectories(path,vName,false))
	{
		ec=0;
		isList.clear();

		for(size_t i=0; i<vName.size(); ++i)
		{
			int idx=_ttoi(vName[i].c_str());
			
			//if is digit string
			if(idx!=0)
			{
				ff::AutoPtr<ISRImages>  isr(new ISRImages);
				if(isr->Create(ff::CatDirectory(path,vName[i])+_T("\\*.*"),3)==0)
				{
					isr->SetID(idx);
					isList.push_back(_ISRPtrT( (ImageSetReader*)isr));
					isr.Detach();
					++ec;
				}
			}
		}
	}

	return ec;
}

const std::string FN_BGDM="bg.dm";
const std::string FN_BGIDX="bg.idx";
const std::string FN_FGIDX="fg.idx";

int CSlippage::Load(const std::string &dir_, double bgWorkPixels)
{
	std::string dir(ff::CatDirectory(dir_,"\\"));

	std::vector<_ISRPtrT> isrList;
	int ec=LoadImageSet(dir+"B",isrList);

	if(ec<=0)
		return -1;
	else
	{
		bgIsrList.swap(isrList);

		bgWorkPixels*=1e6;

		double bgWidth=bgIsrList.front()->Width(), bgHeight=bgIsrList.front()->Height();

		m_bgWorkScale=sqrt(bgWorkPixels/(bgWidth*bgHeight));
		m_bgWorkSize=cv::Size(int(bgWidth*m_bgWorkScale+0.5), int(bgHeight*m_bgWorkScale+0.5));

		for(size_t i=0; i<bgIsrList.size(); ++i)
			bgIsrList[i]->SetDSize(m_bgWorkSize);
	}
	
	ec=LoadImageSet(dir+"F",isrList);
	if(ec!=1)
		return -2;
	else
		fgIsrPtr=isrList.front();

	ec=LoadImageSet(dir+"A",isrList);
	if(ec<=0)
		return -3;
	else
		alphaIsrList.swap(isrList);

	dataDir=dir;

	return 0;
};

cv::Mat CSlippage::GetInitTransform(double bgScale)
{
	cv::Size bgSize(this->GetBgWorkSize()), fgSize(this->GetFgSize());

	double sx(double(fgSize.width)/bgSize.width), sy(double(fgSize.height)/bgSize.height);
	double scale=__max(sx,sy);

	//scale to similar size
	cv::Mat T=(cv::Mat_<double>(3,3)<<scale,0,0, 0,scale,0, 0,0,1);

	Point2f bgCenter(float(bgSize.width*scale/2), float(bgSize.height*scale/2));
	Point2f fgCenter(this->GetFgCenter());

	{//align center
		Point2f dv(fgCenter-bgCenter);
		T=(cv::Mat_<double>(3,3)<<1,0,dv.x, 0,1,dv.y, 0,0,1)*T;
	}

	{//scale with respect to the center
		cv::Mat TS=cv::getRotationMatrix2D(fgCenter,0,bgScale);
		affine_to_homogeneous(TS);
		T=TS*T;
	}

	return T;
}


struct _cc_info
{
	uchar	m_mv;
	int		m_np;
};

static void _get_cc_info(const uchar *mask, int width, int height, int mstep, const int *cc, int cstride, _cc_info *cci, int ncc)
{
	memset(cci,0,sizeof(_cc_info)*ncc);

	for(int yi=0; yi<height; ++yi, mask+=mstep, cc+=cstride)
	{
		for(int xi=0; xi<width; ++xi)
		{
			_cc_info *ccix=&cci[cc[xi]];

			ccix->m_mv=mask[xi];
			ccix->m_np++;
		}
	}
}

static void _remask_alpha(uchar *alpha, int width, int height, int astep, const int *cc, int cstride, const _cc_info *cci, uchar dval)
{
	for(int yi=0; yi<height; ++yi, alpha+=astep, cc+=cstride)
	{
		for(int xi=0; xi<width; ++xi)
		{
			if(cci[cc[xi]].m_mv==1)
				alpha[xi]=dval;
		}
	}
}

void vm_erase_small_cc(uchar *alpha, int width, int height, int astep, uchar T, int ncc_keep)
{
	uchar *mask=new uchar[width*height];
	ff::ipf_threshold(alpha,width,height,astep,mask,width,T+1,T+1,0,0,255);

	int *cc=new int[width*height];
	int ncc=ff::ipf_connected_component(mask,width,height,width,cc,width);

	_cc_info *cci=new _cc_info[ncc];
	_get_cc_info(mask,width,height,width,cc,width,cci,ncc);

	int *vnp=new int[ncc];
	int  nfc=0;
	for(int i=0; i<ncc; ++i)
	{
		if(cci[i].m_mv==255)
		{
			vnp[nfc++]=cci[i].m_np;
		}
	}

	if(nfc>ncc_keep)
	{
		std::sort(vnp,vnp+nfc, std::greater<int>());

		int np_min=vnp[ncc_keep-1];
		for(int i=0; i<ncc; ++i)
		{
			if(cci[i].m_mv==255 && cci[i].m_np<np_min)
				cci[i].m_mv=1;
		}

		_remask_alpha(alpha,width,height,astep,cc,width,cci,0);
	}

	delete[]vnp;
	delete[]cci;
	delete[]cc;
	delete[]mask;
}

void getFootPoints(const cv::Mat &_alpha, cv::Point2f &footPoints, int NROW=10)
{
	float xs=0, ys=0, n=0;
	int nrow=0;

	cv::Mat alpha(_alpha.clone());
	vm_erase_small_cc(_CV_DWHS(alpha),127,1);

	for(int yi=alpha.rows-1; yi>=0; --yi)
	{
		const uchar *px=alpha.ptr<uchar>(yi);
		bool empty=true;

		for(int xi=0; xi<alpha.cols; ++xi, px+=alpha.channels())
		{
			if(*px>127)
			{
				xs+=xi; ys+=yi; n+=1;
				empty=false;
			}
		}

		if(!empty)
		{
			++nrow;
		}

		if(nrow>=NROW)
			break;
	}

	if(n>1)
		footPoints = (cv::Point2f(xs/n,ys/n));
	else
		footPoints = (cv::Point2f(alpha.cols/2,alpha.rows/2));
}

void _filter_points_by_dist(const std::vector<Point2f> &vpt, std::vector<Point2f> &vptx, const std::vector<Point2f> &footPoints, size_t maxNP)
{
	if(vpt.size()<=maxNP)
	{
		vptx=vpt;
		return;
	}

	std::vector<std::pair<float,size_t> > vdist;
	vdist.reserve(vpt.size());

	for(size_t i=0; i<vpt.size(); ++i)
	{
		float minDist=FLT_MAX;

		for(size_t j=0; j<footPoints.size(); ++j)
		{
			Point2f dv(vpt[i]-footPoints[j]);
			float d=dv.dot(dv);
			if(d<minDist)
				minDist=d;
		}
		vdist.push_back(std::pair<float,size_t>(minDist, i));
	}

	std::sort(vdist.begin(), vdist.end());
	
	std::vector<Point2f> vt;
	for(size_t i=0; i<maxNP; ++i)
	{
		vt.push_back(vpt[ vdist[i].second ]);
	}
	vptx.swap(vt);
}

void _filter_points_by_color(const std::vector<Point2f> &vpt, std::vector<Point2f> &vptx, const std::vector<Point2f> &baseSet, const cv::Mat &img, size_t maxNP)
{
	if(vpt.size()<=maxNP)
	{
		vptx=vpt;
		return;
	}

	std::vector<std::pair<float,size_t> > vdist;
	vdist.reserve(vpt.size());

	for(size_t i=0; i<vpt.size(); ++i)
	{
		float minDist=FLT_MAX;

		assert(img.ptr(0,1)==img.data+3);
		const uchar *ci=img.ptr(int(vpt[i].y+0.5),int(vpt[i].x+0.5) );

		for(size_t j=0; j<baseSet.size(); ++j)
		{
			const uchar *cj=img.ptr(int(baseSet[j].y+0.5),int(baseSet[j].x+0.5));
			float diff=(int)ff::px_diff_c3(ci,cj);
			if(diff<minDist)
				minDist=diff;
		}

		vdist.push_back(std::pair<float,size_t>(minDist,i));
	}

	std::sort(vdist.begin(), vdist.end());
//	std::reverse(vdist.begin(),vdist.end());
	std::vector<Point2f> vt;
	for(size_t i=0; i<maxNP; ++i)
	{
		vt.push_back(vpt[ vdist[i].second ]);
	}
	vptx.swap(vt);
}

void _filter_points(std::vector<Point2f> &vpt, const std::vector<Point2f> &footPoints, const cv::Mat &img, size_t maxNP, int meanWndSize=9, size_t baseNP=10)
{
	cv::Mat simg;
	cv::blur(img, simg, cv::Size(meanWndSize,meanWndSize));

	std::vector<Point2f> baseSet;
	_filter_points_by_dist(vpt,baseSet,footPoints, baseNP);

	_filter_points_by_color(vpt,vpt,baseSet,simg,maxNP);
}

void _smooth_points(const std::vector<cv::Point2f> &pt, std::vector<cv::Point2f> &dpt, int W)
{
	if(!pt.empty())
	{
		std::vector<Point2f>  _spt(pt.size()+1);
		_spt[0]=Point2f(0,0);

		Point2f *spt=&_spt[1];
		for(size_t i=0; i<pt.size(); ++i)
		{
			spt[i]=spt[i-1]+pt[i];
		}

		dpt.resize(pt.size());
		for(int i=0; i<(int)pt.size(); ++i)
		{
			int ib=i-W/2, ie=i+W/2;
			int sl=i-__max(0,ib);
			int sr=__min(ie,(int)pt.size()-1)-i;
			int hwsz=__min(sl,sr), wsz=2*hwsz+1;

			Point2f dv(spt[i+hwsz]-spt[i-hwsz-1]);
			dpt[i]=Point2f(dv.x/wsz, dv.y/wsz);
		}
	}
}

void _smooth_foot_points(FgDataSet &fds, int WSZ)
{
	std::vector<Point2f> pt;
	for(int i=0; i<fds.Size(); ++i)
	{
		FgFrameData *fd=fds.GetAtIndex(i);
		pt.push_back(fd->m_FootPoints.front());
	}
	_smooth_points(pt,pt,WSZ);

	for(int i=0; i<fds.Size(); ++i)
	{
		FgFrameData *fd=fds.GetAtIndex(i);
		assert(fd->m_FootPoints.size()>=1);
		fd->m_FootPoints[0]=pt[i];
	}
}

int CSlippage::UpdateFgData(const cv::Mat &featureMask, int footSmoothWSZ,  int nKLTDetect, int nKLTSelect)
{
	FgDataSet fgDataSet;
	if(this->_LoadDataSet(&fgDataSet)<0)
	{
		return -1;
	}

	ImageSetReader &fgIsr(*fgIsrPtr);

	Mat preImg, preAlpha;
	int jstart=0;
	for(int j=jstart; j<fgIsr.Size(); ++j)
	{
		std::cout << "Frame:" << j << endl;
		FgFrameData *fd=fgDataSet.Get(j);
		fd->m_FootPoints.resize(alphaIsrList.size());

		Mat curAlpha;
		for(size_t k=0; k<alphaIsrList.size(); ++k)
		{
			ImageSetReader &alphaIsr(*(alphaIsrList[k]));
			Mat tempAlpha = alphaIsr.Read(j)->clone();
			cv::cvtColor(tempAlpha,tempAlpha,CV_BGR2GRAY);
			cv::resize(tempAlpha, tempAlpha, Size(), m_fgWorkScale, m_fgWorkScale);

			getFootPoints(tempAlpha, fd->m_FootPoints[k]);

			if(k==0)
				curAlpha = tempAlpha.clone();
			else
				bitwise_or(curAlpha, tempAlpha, curAlpha);
		}

		Mat curImg=fgIsr.Read(j)->clone();
		cv::resize(curImg, curImg, Size(), m_fgWorkScale, m_fgWorkScale);

		if(j == jstart)
		{
			fd->m_RelativeAffine = (Mat_<double>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1 );
		}
		else
		{
			Mat erodeAlpha; 
			
			threshold( preAlpha, erodeAlpha, 0.5, 255, THRESH_BINARY_INV );
			Mat element = getStructuringElement( MORPH_RECT, cv::Size(9, 9), cv::Point(4, 4) );
			erode( erodeAlpha, erodeAlpha, element, cv::Point(-1,-1), 2);
			bitwise_and(featureMask, erodeAlpha, erodeAlpha);

			vector<Point2f> prekeyPoints, curKeyPoints;
			int n=getKltkeyPoints( preImg, prekeyPoints, nKLTDetect, erodeAlpha);

	//		showFeatures(preImg,prekeyPoints,0);

	//		_filter_points(prekeyPoints, fd->m_FootPoints, preImg, nKLTSelect);
			
	//		showFeatures(preImg,prekeyPoints,0);

			Mat affine;
			getKltTranslation( preImg, curImg,  prekeyPoints, curKeyPoints, affine);
			fd->m_RelativeAffine = affine;


	//		showMatch(preImg,prekeyPoints,curImg,curKeyPoints,NULL);
	//		showRegist(preImg,curImg,affine,4);	

			cout << affine << endl;
		}

		preImg=curImg;
		preAlpha=curAlpha;
	}

	_smooth_foot_points(fgDataSet, footSmoothWSZ);
	
	return this->_SaveDataSet(&fgDataSet);
}

int CSlippage::_LoadDataSet(FgDataSet *fg, BgDataSet *bg, BgDiskMap *bgm)
{
	int ec=0;
	if(fg)
	{
		ec=fg->Load(dataDir+FN_FGIDX);
	}

	if(ec>=0 && bg)
	{
		ec=bg->Load(dataDir+FN_BGIDX);
	}

	if(ec>=0 && bgm)
	{
		ec=-1;
		try
		{
			bgm->Load(dataDir+FN_BGDM);
			ec=0;
		}
		catch(...){}
	}

	return ec;
}

int CSlippage::_SaveDataSet(FgDataSet *fg, BgDataSet *bg, BgDiskMap *bgm)
{
	int ec=0;
	if(fg)
	{
		ec=fg->Save(dataDir+FN_FGIDX);
	}

	if(ec>=0 && bg)
	{
		ec=bg->Save(dataDir+FN_BGIDX);
	}

	if(ec>=0 && bgm)
	{
		ec=-1;
		try
		{
			bgm->Save();
			ec=0;
		}
		catch(...){}
	}

	return ec;
} 

void getKLTFeaturesInGrids(const cv::Mat &img, std::vector<cv::Mat> &gridMask, std::vector<Point2f> &vfea, int nfeaInEachGrid)
{
	vfea.resize(0);

	std::vector<Point2f> gridFea;
	for(size_t i=0; i<gridMask.size(); ++i)
	{
		gridFea.resize(0);
		getKltkeyPoints(img, gridFea, nfeaInEachGrid, gridMask[i]);

		std::copy(gridFea.begin(), gridFea.end(), std::back_inserter(vfea));
	}
}

void makeGridMask(cv::Size imgSize, std::vector<cv::Mat> &gridMask, cv::Size grid)
{
	double dx=double(imgSize.width)/grid.width, dy=double(imgSize.height)/grid.height;

	cv::Mat mask(imgSize.height, imgSize.width, CV_8UC1);

	for(int yi=0; yi<grid.height; ++yi)
	{
		for(int xi=0; xi<grid.width; ++xi)
		{
			mask=0;
			mask(cv::Rect(int(xi*dx+0.5), int(yi*dy+0.5), int(dx+0.5), int(dy+0.5) ) )=255;
			gridMask.push_back(mask.clone());
		}
	}
}

int CSlippage::UpdateBgData(cv::Size KLTGrid, int nFeaInEachGrid)
{
	BgDataSet bgDataSet;
	BgDiskMap bgDiskMap;

	if(this->_LoadDataSet(NULL,&bgDataSet,&bgDiskMap)<0)
	{
		return -1;
	}

	std::vector<cv::Mat> gridMask;
	makeGridMask(this->GetBgWorkSize(), gridMask, KLTGrid);

	// background
	for(uint i=0; i<bgIsrList.size(); i++)
	{
		ImageSetReader&bgIsr(*bgIsrList[i]);

		for(int j=0; j<bgIsr.Size(); ++j)
		{
			cv::Mat curImg;
			bgIsr.ReadSized(j,curImg);

			vector<KeyPoint> siftKeyPoints;
			Mat siftDescriptor;
			getSiftKeyPoints( curImg, siftKeyPoints, siftDescriptor );
			vector<Point2f> points;
			keyPointsToPoints(siftKeyPoints, points);

			BgFrameData *fd=bgDataSet.Get(make_ufid(i, j));
			fd->m_videoID = i;
			fd->m_FrameID = j;
			fd->m_keyPoints = points;

			BgFrameDataEx dx;
			dx.m_descriptor=siftDescriptor;

			getKLTFeaturesInGrids(curImg, gridMask, dx.m_KLTFeatures, nFeaInEachGrid);

			bgDiskMap.AddData(make_ufid(i,j),dx,true);

			printf("loading %s",bgIsr.FrameName(j).c_str());
			cout << "sift:" << siftKeyPoints.size() << endl;
		}
	}

	return this->_SaveDataSet(NULL,&bgDataSet,&bgDiskMap);
}



struct _Match
{
	int		index;
	float   dst_error;
	float	tsf_error;
public:
	_Match(int _index=0, float _dst_error=0, float _tsf_error=0)
		:index(_index),dst_error(_dst_error),tsf_error(_tsf_error)
	{
	}
	friend bool operator<(const _Match &left, const _Match &right)
	{
		return left.dst_error<right.dst_error;
	}

	DEFINE_BFS_IO_3(_Match,index,dst_error,tsf_error);
};

struct _NNData
{
	std::vector<_Match>	m_nbr;
	
	DEFINE_BFS_IO_1(_NNData,m_nbr);
};

void _add_image_corners(std::vector<cv::Point3f> &vpt, int width, int height)
{
	//arrange counter-clockwise
	vpt.push_back(Point3f(0.0f,0.0f,1.0f));
	vpt.push_back(Point3f(0,(float)height,1));
	vpt.push_back(Point3f((float)width,(float)height,1));
	vpt.push_back(Point3f((float)width,0,1));
}

const int VIMAGE_WIDTH=500, VIMAGE_HEIGHT=500;
void _add_distortion_points(std::vector<Point3f> &vpt)
{
	_add_image_corners(vpt,VIMAGE_WIDTH,VIMAGE_HEIGHT);
}

float _normalize_distortion(float distortion)
{
	return 2*distortion/(VIMAGE_WIDTH+VIMAGE_HEIGHT);
}

double _poly_area(const std::vector<Point3f> &vpt)
{
	double s=0;

	for(size_t i=1; i<vpt.size(); ++i)
	{
		s+=(vpt[i-1].x-vpt[i].x)*(vpt[i-1].y+vpt[i].y);
	}
	s+=(vpt.back().x-vpt[0].x)*(vpt.back().y+vpt[0].y);
	
	return fabs(s*0.5);
}

double _tri_area(const Point3f &p0, const Point3f &p1, const Point3f &p2)
{
	double s=(p0.x-p1.x)*(p0.y+p1.y);
	s+=(p1.x-p2.x)*(p1.y+p2.y);
	s+=(p2.x-p0.x)*(p2.y+p0.y);

	return fabs(s*0.5);
}

float _get_transform_distortion(const cv::Mat &T)
{
	static std::vector<cv::Point3f> vpt;
	if(vpt.empty())
	{
		_add_distortion_points(vpt);
	}

	return _get_transform_distortion(T, vpt);
}

float _get_transform_residual(const cv::Mat &T, const std::vector<cv::Point2f> &pt, const std::vector<cv::Point2f> &ref_pt)
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

	return sqrt(err);
}

bool  getHomography(const std::vector<Point2f> &points1, const std::vector<Point2f> &points2, cv::Mat &H, float *distortion_err, float *transform_err, float *reject_ratio, float RANSAC_ERR)
{
	assert(points1.size()==points2.size());
	if(points1.size()<8)
		return false;

	Mat ransac_mask;
	H = cv::findHomography(points1, points2, CV_RANSAC, RANSAC_ERR, ransac_mask);

	if(distortion_err)
		*distortion_err = _get_transform_distortion(H);

	if(transform_err)
	{
		std::vector<Point2f> goodPoints1, goodPoints2;

		uchar *p = ransac_mask.ptr<uchar>();
		for (unsigned int k=0; k<points1.size(); k++)
		{
			if(p[k] != 0)
			{
				goodPoints1.push_back(points1[k]);
				goodPoints2.push_back(points2[k]);
			}
		}

		if(reject_ratio)
			*reject_ratio=1.0f-float(goodPoints1.size())/points1.size();

		*transform_err = _get_transform_residual(H, goodPoints1, goodPoints2);
	}
	else
	{
		if(reject_ratio)
		{
			int nsel=0;
			uchar *p = ransac_mask.ptr<uchar>();
			for (unsigned int k=0; k<points1.size(); k++)
			{
				if(p[k] != 0)
					++nsel;
			}

			*reject_ratio=1.0f-float(nsel)/points1.size();
		}
	}

	return true;
}

bool  getHomography(const std::vector<Point2f> &points1, const cv::Mat &descriptor1, const std::vector<Point2f> &points2, const cv::Mat &descriptor2, cv::Mat &H, float *distortion_err,float *transform_err, float *reject_ratio, float RANSAC_ERR)
{
	vector<Point2f> mpoints1, mpoints2;
	getSiftMatchNoRansac(descriptor1, descriptor2, points1, points2, mpoints1, mpoints2);

	return getHomography(mpoints1,mpoints2,H,distortion_err,transform_err, reject_ratio, RANSAC_ERR);
}

void knn_refine(BgDataSet &bgDataSet, BgDiskMap &bgDiskMap, Mat &descriptor, const std::vector<Point2f> &points, const std::vector<_Match> &nbr, std::vector<_Match> &dnbr, const int K, const int D, const float MAX_DST_ERR, const float MAX_TSF_ERR, const float MAX_REJECT, const float RANSAC_ERR)
{
	std::vector<char>  mask(bgDataSet.Size(),0);

	for(uint i=0; i<nbr.size(); ++i)
	{
		int ib=nbr[i].index-D/2, ie=nbr[i].index+D/2;
		ib=__max(0,ib);
		ie=__min(int(bgDataSet.Size()),ie);

		for(int j=ib; j<ie; ++j)
		{
			if(mask[j]==0)
			{
				BgFrameData *curFd=bgDataSet.GetAtIndex(j);
				BgDiskMap::AutoDataPtr aptr(bgDiskMap.GetData(make_ufid(curFd->m_videoID,curFd->m_FrameID),true));

				float distortion_err, transform_err, reject_ratio;
				cv::Mat H;
				if(getHomography(points,descriptor,curFd->m_keyPoints,aptr->m_descriptor,H,&distortion_err, &transform_err, &reject_ratio, RANSAC_ERR))
				{
					if(distortion_err < MAX_DST_ERR && transform_err < MAX_TSF_ERR && reject_ratio<MAX_REJECT || reject_ratio<0.63 )
					{
						dnbr.push_back(_Match(j,distortion_err, transform_err));
					}
				}
			
				mask[j]=1;
			}
		}
	}

	if((int)(dnbr.size())>K)
	{
		std::sort(dnbr.begin(),dnbr.end());
		dnbr.resize(K);
	}
}

void knn_search(BgDataSet &bgDataSet, BgDiskMap &bgDiskMap, std::vector<_NNData> &nn, CSlippage::KNNParam param)
{
	nn.resize(bgDataSet.Size());

	for(uint i=0; i<bgDataSet.Size(); i+=param.D)
	{
		BgFrameData *fd=bgDataSet.GetAtIndex(i);
		
		BgDiskMap::AutoDataPtr aiptr(bgDiskMap.GetData(make_ufid(fd->m_videoID,fd->m_FrameID),true));
		Mat &descriptor = aiptr->m_descriptor;

		vector<Point2f> &points(fd->m_keyPoints);
		if(points.size() < 8) continue;

		std::vector<_Match> vmatch;
		vmatch.reserve(bgDataSet.Size());

		float d_min=FLT_MAX, t_min=FLT_MAX, r_min=FLT_MAX;

		for(uint j=0; j<bgDataSet.Size(); j+=param.D)
		{
			if(abs((int)i-(int)j)>param.INTERNEL_EXCLUDE_RANGE) 
			{
				BgFrameData *curFd=bgDataSet.GetAtIndex(j);
				BgDiskMap::AutoDataPtr ajptr(bgDiskMap.GetData(make_ufid(curFd->m_videoID,curFd->m_FrameID),true));

				float distortion_err, transform_err, reject_ratio;
				cv::Mat H;
				
				if(getHomography(points,descriptor,curFd->m_keyPoints,ajptr->m_descriptor,H,&distortion_err,&transform_err, &reject_ratio, 10))
				{
			//		if(distortion_err < param.MAX_DST_ERR && transform_err < param.MAX_TSF_ERR && reject_ratio<param.MAX_REJECT_RATIO)
			//		if(distortion_err < 0.05 && transform_err < 16 && reject_ratio<0.75f)
					if(reject_ratio<0.75)
					{
			//			printf("\n%f %f %f", distortion_err, transform_err, reject_ratio);

						vmatch.push_back(_Match(j,distortion_err,transform_err));
					}

					if(distortion_err<d_min) d_min=distortion_err;
					if(transform_err<t_min) t_min=transform_err;
					if(reject_ratio<r_min) r_min=reject_ratio;
				}
			}
		}

		printf("\nmin=%f %f %f",d_min,t_min,r_min);

		std::sort(vmatch.begin(),vmatch.end());
		if(int(vmatch.size())>param.K0)
			vmatch.resize(param.K0);

		std::cout<<"\nL0:"<<i<<" nm="<<vmatch.size()<<endl;
		for(uint j=0; j<vmatch.size(); ++j)
		{
			printf("\n%d-%d",i,vmatch[j].index);
		}

		nn[i].m_nbr.swap(vmatch);
	}

	std::vector<_Match> nbr0;

	for(uint i=0; i<bgDataSet.Size(); ++i)
	{
		BgFrameData *fd=bgDataSet.GetAtIndex(i);
		
		BgDiskMap::AutoDataPtr aiptr(bgDiskMap.GetData(make_ufid(fd->m_videoID,fd->m_FrameID),true));
		Mat &descriptor = aiptr->m_descriptor;

		if(i%param.D==0)
		{
			int ib=i/param.D*param.D, ie=(i/param.D+1)*param.D;
			nbr0=nn[ib].m_nbr;
			if((size_t)ie<bgDataSet.Size())
			{
				std::copy(nn[ie].m_nbr.begin(),nn[ie].m_nbr.end(),std::back_inserter(nbr0));
			}
		}

		vector<Point2f> &points(fd->m_keyPoints);
		if(points.size() < 8) continue;
		
		if(!nbr0.empty())
			knn_refine(bgDataSet,bgDiskMap,descriptor,points,nbr0,nn[i].m_nbr,param.K1,param.D,param.MAX_DST_ERR, param.MAX_TSF_ERR, param.MAX_REJECT_RATIO, param.RANSAC_ERR);

		std::cout<<"L1:"<<i<<" nm0="<<nbr0.size()<<" nm="<<nn[i].m_nbr.size()<<endl;
	}
}

void _allocGridPoints(const std::vector<Point2f> &vfea, std::vector<std::vector<int> > &gridFea, const std::vector<Point2f> &gridCenters, int minPointsInEachGrid)
{
	gridFea.resize(gridCenters.size());
	for(size_t i=0; i<gridFea.size(); ++i)
		gridFea[i].resize(0);

	for(size_t i=0; i<vfea.size(); ++i)
	{
		float dmin=FLT_MAX;
		int   imin=-1;

		for(size_t j=0; j<gridCenters.size(); ++j)
		{
			cv::Point2f dv(gridCenters[j]-vfea[i]);
			float dist=dv.dot(dv);
			if(dist<dmin)
			{
				dmin=dist; imin=(int)j;
			}
		}

		gridFea[imin].push_back(i);
	}

	minPointsInEachGrid=__min(minPointsInEachGrid, (int)vfea.size());

	std::vector<std::pair<float,int> > vsort;
	vsort.reserve(vfea.size());

	for(size_t i=0; i<gridFea.size(); ++i)
	{
		if(gridFea[i].size()<minPointsInEachGrid)
		{
			vsort.resize(0);

			for(size_t j=0; j<vfea.size(); ++j)
			{
				cv::Point2f dv(gridCenters[i]-vfea[j]);
				float dist=dv.dot(dv);
				vsort.push_back( std::pair<float,int> (dist, (int)j) );
			}

			std::sort(vsort.begin(), vsort.end());

			gridFea[i].resize(0);
			for(size_t j=0; j<minPointsInEachGrid; ++j)
			{
				gridFea[i].push_back(vsort[j].second);
			}
		}
	}
}

int calcSubHomographies(std::vector<cv::Mat> &subH, cv::Mat &refinedH, const cv::Mat &imi, const cv::Mat &imj, const std::vector<Point2f> &iKLT, const cv::Mat &Hij, float ransacThreshold, const std::vector<Point2f> &gridCenters, int minPointsInEachGrid, float *reject_ratio)
{
	cv::Mat warpImg, warpMask;

	cv::warpPerspective(imi,warpImg,Hij,imi.size(),cv::INTER_LINEAR,BORDER_REFLECT);

	cv::Mat mask(imi.rows, imi.cols, CV_8UC1);
	mask=255;

	cv::warpPerspective(mask,warpMask,Hij,imi.size(),cv::INTER_NEAREST);

	std::vector<Point2f> iKeyPoints(iKLT.size());
	do_transform(Hij, &iKLT[0], Point2f(0,0), &iKeyPoints[0], int(iKLT.size()));

	std::vector<Point2f> iMatch, jMatch;
	KLTTrackNoRansac(warpImg,imj,iKeyPoints,iMatch,jMatch,3);

	if(iMatch.size()<16)
	{
		refinedH=Hij;
		return -1;
	}

//	printf("\nnfea1=%d",iMatch.size());

	std::vector<Point2f> iGood, jGood;
	Mat Hx = findHomographyEx(iMatch, jMatch, CV_RANSAC, ransacThreshold, &iGood, &jGood);

	if(reject_ratio)
		*reject_ratio=1.0f-float(iGood.size())/iMatch.size();

//	showMatch(imi,iMatch,imj,jMatch,NULL);

//	printf("\nnfea2=%d",iGood.size());

	refinedH=Hx*Hij;

#if 0 //not computed for the case of main planar ground, use the @bgMask param to roughly speciy the region of ground can well deal with most situations.

	cv::Mat Hji(Hij.inv());
	iMatch.resize(iGood.size());
	do_transform(Hji,&iGood[0],Point2f(0,0),&iMatch[0],(int)iGood.size());

	std::vector<std::vector<int> > gridFea;
	_allocGridPoints(iMatch,gridFea,gridCenters,minPointsInEachGrid);

	do_transform(Hx,&iGood[0],Point2f(0,0),&iGood[0],(int)iGood.size());

	subH.resize(0);
	for(size_t i=0; i<gridFea.size(); ++i)
	{
		iMatch.resize(0);
		jMatch.resize(0);

		for(size_t j=0; j<gridFea[i].size(); ++j)
		{
			int index=gridFea[i][j];
			iMatch.push_back(iGood[ index ]);
			jMatch.push_back(jGood[ index ]);
		}

	//	showMatch(imi,iMatch,imj,jMatch,NULL);

	//	printf("\nisize=%d",iMatch.size());

		cv::Mat Hi=::findHomography(iMatch,jMatch,0);
	//	cv::Mat Hi=::findHomography(iGood,jGood,CV_RANSAC);
		subH.push_back(Hi*refinedH);
	}
#endif

	return 0;
}

void calcGridCenters(cv::Size imgSize, cv::Size gridSize, std::vector<Point2f> &gridCenters)
{
	float dx=float(imgSize.width)/gridSize.width, dy=float(imgSize.height)/gridSize.height;

	gridCenters.resize(0);
	for(int yi=0; yi<gridSize.height; ++yi)
	{
		for(int xi=0; xi<gridSize.width; ++xi)
		{
			gridCenters.push_back(Point2f(dx*xi+dx*0.5f, dy*yi+dy*0.5f));
		}
	}
}

void selectFeatures(const std::vector<Point2f> &vsrc, std::vector<Point2f> &vsel, const cv::Mat &mask)
{
	vsel.resize(0);
	for(size_t i=0; i<vsrc.size(); ++i)
	{
		if(mask.at<char>((int)vsrc[i].y, (int)vsrc[i].x)!=0)
			vsel.push_back(vsrc[i]);
	}
}

int CSlippage::UpdateBgKNN(KNNParam param, bool updateNN, bool findFarNbrs, bool updateTransform, const cv::Mat &feaMask)
{
	const double globalRansacThreshold=2.0;

	BgDataSet bgDataSet;
	BgDiskMap bgDiskMap;

	if(this->_LoadDataSet(NULL,&bgDataSet,&bgDiskMap)<0)
	{
		return -1;
	}

	std::vector<_NNData> vnn;
	if(updateNN && findFarNbrs)
		knn_search(bgDataSet,bgDiskMap,vnn, param);

	std::vector<cv::Point2f> gridCenters;
	calcGridCenters(this->GetBgWorkSize(), param.gridSize, gridCenters);

	int istart=0;//94;

	std::vector<Point2f> _selKLT, *selKLT=NULL;

	for(uint i=istart; i<bgDataSet.Size(); ++i)
	{
		BgFrameData *fd=bgDataSet.GetAtIndex(i);

		printf("\nvid=%d, fid=%d",fd->m_videoID, fd->m_FrameID);

		if(updateNN)
		{
			fd->m_nbr.clear();

			BgFrameData::Neighbor nbr;
		
			// add forward and backward 5 frames
			for(int curFrameId=fd->m_FrameID-param.INTERNEL_INCLUDE_RANGE; curFrameId<=fd->m_FrameID+param.INTERNEL_INCLUDE_RANGE; curFrameId++)
			{
				if(curFrameId < 0 || curFrameId >= bgIsrList[fd->m_videoID]->Size() ) 
					continue;

				nbr.m_index=bgDataSet.GetIndex(make_ufid(fd->m_videoID, curFrameId));
				fd->m_nbr.push_back(nbr);
			}

			fd->m_nTemporalNBR=(int)fd->m_nbr.size();

			if(findFarNbrs)
			{
				for(size_t j=0; j<vnn[i].m_nbr.size(); ++j)
				{
					nbr.m_index=vnn[i].m_nbr[j].index;
					fd->m_nbr.push_back(nbr);
				}
			}
		}

		if(updateTransform)
		{
			cv::Mat imi;
			bgIsrList[fd->m_videoID]->ReadSized(fd->m_FrameID,imi);
			BgDiskMap::AutoDataPtr aiptr(bgDiskMap.GetData(make_ufid(fd->m_videoID,fd->m_FrameID),true));
			Mat &descriptor = aiptr->m_descriptor;
			vector<Point2f> &points_sift (fd->m_keyPoints);

			if(feaMask.data)
			{
				selectFeatures(aiptr->m_KLTFeatures, _selKLT, feaMask);
				selKLT=&_selKLT;
			}
			else
				selKLT=&aiptr->m_KLTFeatures;


			for(size_t k=0; k<fd->m_nbr.size(); ++k)
		//	for(size_t k=fd->m_nTemporalNBR; k<fd->m_nbr.size(); ++k)
			{
				BgFrameData *curFd=bgDataSet.GetAtIndex(fd->m_nbr[k].m_index);

				BgDiskMap::AutoDataPtr ajptr(bgDiskMap.GetData(make_ufid(curFd->m_videoID,curFd->m_FrameID),true));

				BgFrameData::Neighbor &neighbor(fd->m_nbr[k]);

				Mat H=(cv::Mat_<double>(3,3)<<1,0,0,0,1,0,0,0,1);
				float distortion_err=0, transform_err=0, reject_ratio=0;

				if(descriptor.size().height>8 && ajptr->m_descriptor.size().height>8)
				{
					vector<Point2f> matched_points1, matched_points2;
					getSiftMatchNoRansac(descriptor, ajptr->m_descriptor, points_sift, curFd->m_keyPoints, matched_points1, matched_points2);

					getHomography(matched_points1,matched_points2,H,NULL,NULL,NULL, param.RANSAC_ERR);
				}
				
				if(H.data)
				{
					cv::Mat imj;
					bgIsrList[curFd->m_videoID]->ReadSized(curFd->m_FrameID,imj);
					BgDiskMap::AutoDataPtr ajptr(bgDiskMap.GetData(make_ufid(curFd->m_videoID,curFd->m_FrameID),true));

			//		printf("\n%d:%d",fd->m_nbr[k].m_index,i);

					Mat refinedH=H;
					calcSubHomographies(fd->m_nbr[k].m_subH, refinedH, imi,imj,*selKLT,H,globalRansacThreshold,gridCenters,param.minPointsInEachGrid,&reject_ratio);
					distortion_err=_get_transform_distortion(refinedH);

			//		printf("\nreject=%f %f",reject_ratio, distortion_err);

			//		if(i==94 && fd->m_nbr[k].m_index== 97)
			//		if(reject_ratio<0.6||distortion_err<0.1)
			//			showRegist(imi,imj,refinedH,8);

					neighbor.m_T = refinedH;
					neighbor.m_distortion_err = distortion_err;
					neighbor.m_transform_err = reject_ratio;
#if 0
				//	cv::Mat imi=bgIsrList[fd->m_videoID]->Read(fd->m_FrameID)->clone();
				//	cv::Mat imj=bgIsrList[curFd->m_videoID]->Read(curFd->m_FrameID)->clone();
					printf("\n#d=%f t=%f r=%f",distortion_err,transform_err,reject_ratio);
					showRegist(imi,imj,T,6);
#endif
				}
			}
		}
	}

	return this->_SaveDataSet(NULL,&bgDataSet,&bgDiskMap);
}


namespace {

double _get_distortion(const cv::Point2f dpt[4])
{
	//must the four corners arranged counter clockwise
	float derr=fabs(dpt[0].x-dpt[1].x)+fabs(dpt[2].x-dpt[3].x)+fabs(dpt[0].y-dpt[3].y)+fabs(dpt[1].y-dpt[2].y);
	derr/=4;

	Point2f dv1( dpt[2]-dpt[0] ), dv2(dpt[3]-dpt[1]);
	float L=(sqrt(dv1.dot(dv1))+sqrt(dv2.dot(dv2)))/2;

	return derr*100/L;
}

}

int CSlippage::StitchBg()
{
	const double MAX_DISTORTION=5;

	BgDataSet bgDataSet;

	if(this->_LoadDataSet(NULL,&bgDataSet,NULL)<0)
	{
		return -1;
	}

	std::vector<cv::Point2f>  vcorners;

	cv::Size  bgSize(this->GetBgWorkSize());
	cv::Point2f  bgCorners[]={Point2f(0,0),Point2f(0,bgSize.height),Point2f(bgSize.width,bgSize.height),Point2f(bgSize.width,0)}, bgCornersWarped[4];

	for(int i=0; i<4; ++i)
		vcorners.push_back(bgCorners[i]);

	std::vector<cv::Point>    points;
	points.resize(4);

	for(size_t k=0; k<bgDataSet.Size(); ++k)
	{
		printf("\nstitch %d",k);

		BgFrameData *bd=bgDataSet.GetAtIndex(k);

		vcorners.resize(4);	

		for(size_t i=0; i<bd->m_nbr.size(); ++i)
		{
			cv::Matx33d HInv( cv::Mat(bd->m_nbr[i].m_T.inv()) );
			do_transform(HInv,bgCorners,Point2f(0,0),bgCornersWarped,4);

			if(_get_bg_distortion(bgCornersWarped)<MAX_DISTORTION)
			{
				for(int j=0; j<4; ++j)
					vcorners.push_back(bgCornersWarped[j]);

				bd->m_nbr[i].m_flag|=BgFrameData::NF_STITCH_CAND;
			}
			else
				bd->m_nbr[i].m_flag&=~BgFrameData::NF_STITCH_CAND;
		}

		cv::Rect bb(_get_bounding_box(vcorners));
		cv::Mat mask(bb.height,bb.width,CV_8UC1);
		mask=0;

		for(size_t i=0; i<vcorners.size(); i+=4)
		{
			for(int j=0; j<4; ++j)
			{
				points[j]=cv::Point(vcorners[i+j].x-bb.x, vcorners[i+j].y-bb.y);
			}

			cv::fillConvexPoly(mask,points,cv::Scalar(255,255,255));
		}

		std::vector<std::vector<cv::Point> > vcont;
		cv::findContours(mask,vcont,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);

		int imax=0;
		if(vcont.size()>1)
		{
			size_t mnp=0;
			for(size_t j=0; j<vcont.size(); ++j)
			{
				if(vcont[j].size()>mnp)
				{
					mnp=vcont[j].size();
					imax=(int)j;
				}
			}
		}

		std::vector<cv::Point> cont;
		cv::approxPolyDP(vcont[imax], cont, 3, true);

		for(size_t i=0; i<cont.size(); ++i)
		{
			cont[i]+=bb.tl();
		}

		bd->m_polyRegion.clear();
		for(size_t i=0; i<cont.size(); ++i)
			bd->m_polyRegion.push_back(cont[i]);

#if 0
		vcont[0]=cont;
		cv::drawContours(mask,vcont,0,Scalar(128,0,0),3);

		printf("\nnp=%d",cont.size());

		cv::rectangle(mask,cv::Rect(-bb.x,-bb.y,bgSize.width,bgSize.height),cv::Scalar(128,0,0));
		imshow("mask",mask);
		cv::waitKey();
#endif

	}
	
	return this->_SaveDataSet(NULL,&bgDataSet,NULL);
}

struct DPNode
{
public:
	int		m_father;
	float	m_error;

	cv::Point3f  m_dstError;
};

struct _DPTNode
{
	cv::Mat	m_A;
	std::vector<cv::Point3f>   m_vpt;
};

void   init_cvt_points(std::vector<cv::Point3f> &vpt)
{
	vpt.push_back(Point3f(0,0,1));
	vpt.push_back(Point3f(-43,-50,1));
	vpt.push_back(Point3f(43,-50,1));
	vpt.push_back(Point3f(-43,25,1));
	vpt.push_back(Point3f(43,25,1));
}

void  trans_cvt_points(const std::vector<cv::Point3f> &vpt, const cv::Point2f &offset, std::vector<cv::Point3f> &dest)
{
	for(size_t i=0; i<vpt.size(); ++i)
	{
		dest.push_back(cv::Point3f(vpt[i].x+offset.x,vpt[i].y+offset.y,1));
	}
}

cv::Mat get_transform(const cv::Mat &T, const std::vector<cv::Point3f> &vpt, float *error=NULL, bool rotation=false)
{
	std::vector<cv::Point3f> ptx;
	cv::transform(vpt,ptx,T);

	cv::Mat TX;
	std::vector<cv::Point2f> pt1,pt2;
	for(size_t i=0; i<vpt.size(); ++i)
	{
		pt1.push_back(Point2f(vpt[i].x/vpt[i].z,vpt[i].y/vpt[i].z));
		pt2.push_back(Point2f(ptx[i].x/ptx[i].z,ptx[i].y/ptx[i].z));
	}

	if(rotation)
		TX=get_tsr_matrix(pt1,pt2);
	else
		TX=get_ts_matrix(pt1,pt2);

	affine_to_homogeneous(TX);

	if(error)
	{
		cv::transform(pt1,ptx,TX);
		*error=0;
		for(size_t i=0; i<pt1.size(); ++i)
		{
			cv::Point2f dv(ptx[i].x-pt2[i].x, ptx[i].y-pt2[i].y);
			*error+=dv.dot(dv);
		}
		*error/=pt1.size();
	}

	return TX;
}

float _get_distortion(const std::vector<cv::Point3f> &bpt, const std::vector<cv::Point3f> &fpt)
{
	double bg_quad_area=_poly_area(bpt);
	double t_error=0;
	for(size_t i=0; i<fpt.size(); ++i)
	{
		double pt_area=_tri_area(fpt[i],bpt[0],bpt[1])+_tri_area(fpt[i],bpt[1],bpt[2])+_tri_area(fpt[i],bpt[2],bpt[3])+_tri_area(fpt[i],bpt[3],bpt[0]);
		t_error+=fabs(pt_area-bg_quad_area);
	}
	t_error*=0.25;

	return (float)t_error;
}

float _get_smooth_error(const std::vector<cv::Point3f> &pt0, const std::vector<cv::Point3f> &pt1)
{
	float error=0;
	for(size_t i=0; i<pt0.size(); ++i)
	{
		cv::Point3f dv(pt0[i]-pt1[i]);
		error+=dv.dot(dv);
	}
	return error;
}

void _do_transform(const cv::Mat &T, const std::vector<cv::Point3f> &ipt, std::vector<cv::Point3f> &dpt)
{
	cv::transform(ipt,dpt,T);

	for(size_t i=0; i<dpt.size(); ++i)
	{
		dpt[i]*=1.0/dpt[i].z;
	}
}

void dp_search_v2(const std::vector<FgFrameData*> &fg,  const std::vector<BgFrameData*> &bg, const cv::Mat &bgT, cv::Size bgSize, cv::Size fgSize, std::vector<int> &path, CSlippage::DPParam param)
{
	const int NM=param.NM;

	std::vector<std::vector<DPNode> >  vdp(fg.size());
	std::vector<_DPTNode>	vA(bg.size()*NM), vAT(bg.size()*NM);

	cv::Mat mI=(cv::Mat_<double>(3,3) <<1, 0, 0, 0, 1, 0, 0, 0, 1);

	std::vector<cv::Point3f> vpt;
	vpt.push_back(Point3f(0.0f,0.0f,1.0f));
	vpt.push_back(Point3f(0,(float)bgSize.height,1));
	vpt.push_back(Point3f((float)bgSize.width,(float)bgSize.height,1));
	vpt.push_back(Point3f((float)bgSize.width,0,1));


	std::vector<cv::Point3f> fpt;
	{
		float dx=0, dy=0;
		fpt.push_back(Point3f(dx,dy,1.0f));
		fpt.push_back(Point3f(dx,dy+fgSize.height,1.0f));
		fpt.push_back(Point3f(dx+fgSize.width, dy+fgSize.height,1.0f));
		fpt.push_back(Point3f(dx+fgSize.width, dy, 1.0f));
	}

	std::vector<cv::Point3f> tpt;
	init_cvt_points(tpt);

	vdp[0].resize(bg.size()*NM);
	for(size_t i=0; i<bg.size()*NM; ++i)
	{
		vdp[0][i].m_father=-1;
		vdp[0][i].m_error=0;

		vA[i].m_A=mI.clone();
		vA[i].m_vpt=vpt;
	}

	cv::Mat bgTInv(bgT.inv());

	for(size_t fi=1; fi<fg.size(); ++fi)
	{
		std::cout<<"dp:"<<fi<<endl;

		vdp[fi].resize(bg.size()*NM);
		
		cv::Mat fT=fg[fi]->m_RelativeAffine;
		affine_to_homogeneous(fT);


		for(size_t bix=0; bix<bg.size()*NM; ++bix)
		{
			const int dbi=bix/NM, dmi=bix%NM;
			BgFrameData *bd=bg[dbi];

			float min_err=1e9;
			int   father=-1;
			Point3f min_dst;
			

			for(int mi=0; mi<NM; ++mi)
			{
				for(size_t k=0; k<bd->m_nbr.size(); ++k)
				{
					if(bd->m_nbr[k].m_transform_err>param.MAX_TSF_ERROR||bd->m_nbr[k].m_distortion_err>param.MAX_DST_ERROR)
						continue;

					int ni=bd->m_nbr[k].m_index*NM+mi;

					Mat cur_m_T = bd->m_nbr[k].m_T;

					if(cur_m_T.data)
					{
						cv::Mat Tk=fT*vA[ni].m_A*bgT*cur_m_T;

						std::vector<cv::Point3f>  tptx,tpty;
						trans_cvt_points(tpt,fg[fi]->m_FootPoints.front(), tptx);
						cv::transform(tptx,tpty,Tk.inv());

						float t_error=0;
				/*		if(dmi==0||dmi==1&&!param.USE_HOMOGRAPH)
						{
							Tk=get_transform(Tk,tpty,&t_error,dmi==0? false : true);
							t_error=param.WEIGHT_TRANSFORM*t_error;
						}*/

						std::vector<Point3f> ptx;
						_do_transform(Tk,vpt,ptx);
						float d_error=param.WEIGHT_DISTORTION *_get_distortion(ptx,fpt);
						float s_error=param.WEIGHT_SMOOTH * _get_smooth_error(ptx,vA[ni].m_vpt);
						
						float error=d_error+s_error+t_error;
						error+=vdp[fi-1][ni].m_error;

						if(error<min_err)
						{
							min_err=error;
							father=ni;
							vAT[bix].m_A=Tk*bgTInv;
							vAT[bix].m_vpt=ptx;

							min_dst=Point3f(d_error,s_error,t_error);
						}
					}
				}
			}

			if(bix==0)
				cout<<vAT[0].m_A<<endl;

			vdp[fi][bix].m_father=father;
			vdp[fi][bix].m_error=min_err;

			vdp[fi][bix].m_dstError=min_dst;
		}

		vA.swap(vAT);
	}

	float err_min=FLT_MAX;
	int imin=0;
	for(size_t i=0; i<bg.size()*NM; ++i)
	{
		if(vdp[fg.size()-1][i].m_error<err_min)
		{
			imin=(int)i;
			err_min=vdp[fg.size()-1][i].m_error;
		}
	}

	printf("\nmin err=%f",err_min);

	Point3f merr(0,0,0);

	path.resize(fg.size());
	path[fg.size()-1]=imin;
	for(size_t i=fg.size()-1; i>0; --i)
	{
		Point3f erri=vdp[i][imin].m_dstError;
		merr+=vdp[i][imin].m_dstError;
		imin=vdp[i][imin].m_father;
		path[i-1]=imin;

		printf("\n%d:%d dst=%.2f,%.2f,%.2f",i-1,imin, erri.x, erri.y, erri.z);
	}

	printf("\nmean error=%.2f\t%.2f\t%.2f", merr.x, merr.y, merr.z);
//	system("pause");
}

void dp_search(FgDataSet &fg,  BgDataSet &bg, const cv::Mat &bgT, cv::Size bgSize, cv::Size fgSize, std::vector<int> &path, CSlippage::DPParam param)
{
	std::vector<FgFrameData*> vfg;
	for(size_t i=0; i<fg.Size(); ++i)
	{
		vfg.push_back(fg.GetAtIndex(i));
	}

	std::vector<BgFrameData*> vbg;
	for(size_t i=0; i<bg.Size(); ++i)
	{
		vbg.push_back(bg.GetAtIndex(i));
	}

	dp_search_v2(vfg,vbg,bgT, bgSize,fgSize,path, param);
}

void _dp_search(FgDataSet &fg, std::vector<TShape> &shape,  BgDataSet &bg, const cv::Mat &bgT, cv::Size bgSize, cv::Size fgSize, CSlippage::DPParam param);

std::vector<TShape> g_shape;

int CSlippage::UpdateCorrespondence(const cv::Mat &BGT, DPParam param)
{
	FgDataSet fgDataSet;
	BgDataSet bgDataSet;

	if(this->_LoadDataSet(&fgDataSet,&bgDataSet)<0)
		return -1;

	fgDataSet.SetStartEndPos(param.foreground_start_end[0], param.foreground_start_end[1]);

	_dp_search(fgDataSet,g_shape,bgDataSet,BGT, this->GetBgWorkSize(), this->GetFgSize(), param);

	//for(size_t i=0; i<path.size(); ++i)
	//{
	//	FgFrameData *curFd = fgDataSet.GetAtIndex(i);
	//	curFd->m_Correspondence = path[i]/param.NM;
	//	curFd->m_TransformMethod=path[i]%param.NM==0? 0 : param.USE_HOMOGRAPH? 2 : 1;
	//}

	return this->_SaveDataSet(&fgDataSet,&bgDataSet);
}

int  CSlippage::SolveBGTransform(const cv::Mat &BGT, DPParam param)
{
	FgDataSet fgDataSet;
	BgDataSet bgDataSet;

	if(this->_LoadDataSet(&fgDataSet, &bgDataSet)<0)
		return -1;

	fgDataSet.SetStartEndPos(param.foreground_start_end[0], param.foreground_start_end[1]);

	void dp_solve(FgDataSet &fg,  BgDataSet &bg, double bgWorkScale, const cv::Mat &_BGT);

	dp_solve(fgDataSet,bgDataSet,m_bgWorkScale,BGT);

	return this->_SaveDataSet(&fgDataSet,&bgDataSet);
}

int init_corners(std::vector<cv::Point2f> &points, cv::Size size)
{
	points.clear();
	int width = size.width;
	int height = size.height;

	points.push_back(Point2f(0, 0));
	points.push_back(Point2f(width, 0));
	points.push_back(Point2f(width, height));
	points.push_back(Point2f(0, height));

	return 0;
}

int CSlippage::GetShadowHomography(const std::vector<Point2f> &transformed_points, vector< vector<Mat> > &vH, DPParam param)
{
	FgDataSet fgDataSet;
	if(this->_LoadDataSet(&fgDataSet)<0)
	{
		return -1;
	}

	fgDataSet.SetStartEndPos(param.foreground_start_end[0], param.foreground_start_end[1]);

	assert(transformed_points.size() == 4);

	Size fg_size = GetFgSize();
	int fg_width = fg_size.width;
	int fg_height = fg_size.height;

	vector<Point2f> src_points, dst_points;
	init_corners(src_points, fg_size);

	for(size_t i=0; i<transformed_points.size(); ++i)
	{
		Point2f point = transformed_points[i];
		float x = point.x * fg_width;
		float y = point.y * fg_height;

		dst_points.push_back(Point2f(x, y));
	}

	Mat H = cv::getPerspectiveTransform(src_points, dst_points);

	/////////////////////////////////////////////////////////////////////////

	vH.clear();
	vH.resize(fgDataSet.Size());

	for(uint i=0; i<fgDataSet.Size(); ++i)
	{
		for(size_t j=0; j<alphaIsrList.size(); ++j)
		{
			Point2f foot_point = fgDataSet.GetAtIndex(i)->m_FootPoints[j];
			Point2f foot_point_trans;

			vector<Point2f> v_foot, v_foot_t;
			v_foot.push_back(foot_point);
			v_foot_t.push_back(foot_point_trans);

			perspectiveTransform(v_foot, v_foot_t, H);

			float dx = v_foot[0].x - v_foot_t[0].x;
			float dy = v_foot[0].y - v_foot_t[0].y;

			Mat T = (Mat_<double>(3, 3) << 1, 0, dx, 0, 1, dy, 0, 0, 1);

			Mat TH = T*H;

			vH[i].push_back(TH);
		}
	}

	return 0;
}

int _synthesis_shadow(Mat &img, Mat shadow, float level)
{
	if( img.channels() != 3)
	{
		return -1;
	}

	if( shadow.channels() != 1 )
	{
		cvtColor(shadow, shadow ,CV_BGR2GRAY); 
	}

	int rows = img.rows;
	int cols = img.cols;
	int rows2 = shadow.rows;
	int cols2 = shadow.cols;

	if( cols > cols2 )
	{
		copyMakeBorder( shadow, shadow, 0, 0, 0, cols-cols2, BORDER_CONSTANT);
	}
	if( rows > rows2 )
	{
		copyMakeBorder( shadow, shadow, 0, rows-rows2, 0, 0, BORDER_CONSTANT);
	}

	blur(shadow, shadow, Size(13,13));

	for( int i = 0; i < rows; ++i )
	{
		uchar* p_img = img.ptr<uchar>(i);
		uchar* p_shadow = shadow.ptr<uchar>(i);
		
		for( int j = 0; j < cols; ++j )
		{
			p_img[3 * j + 0] = (p_img[3 * j + 0] * (255 - p_shadow[j] * level)) / 255;
			p_img[3 * j + 1] = (p_img[3 * j + 1] * (255 - p_shadow[j] * level)) / 255;
			p_img[3 * j + 2] = (p_img[3 * j + 2] * (255 - p_shadow[j] * level)) / 255;
		}
	}

	return 0;
}

int _calculate_below_shadow(const Mat &alpha, Mat &shadow_alpha)
{
	Mat _alpha = alpha.clone();
	shadow_alpha = Mat::zeros(_alpha.size(), CV_8U);
	int rows = shadow_alpha.rows;
	int cols = shadow_alpha.cols;

	if( _alpha.channels() != 1 )
	{
		cvtColor(_alpha, _alpha ,CV_BGR2GRAY); 
	}

	_alpha = _alpha > 0;

	Mat element = getStructuringElement( MORPH_RECT, cv::Size(31, 31), cv::Point(15, 15) );
	morphologyEx(_alpha, _alpha, MORPH_OPEN, element);
	morphologyEx(_alpha, _alpha, MORPH_CLOSE, element);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(_alpha.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	int idx = 0, largestComp = 0;
    double minArea = 100;
	for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        const vector<Point>& c = contours[idx];
        double area = fabs(contourArea(Mat(c)));
        if( area > minArea )
        {
            Rect bounding_rect = boundingRect(c);

			int down_point_x = bounding_rect.x + bounding_rect.width/2;
			int down_point_y = bounding_rect.y + bounding_rect.height;

			float radius = __min(bounding_rect.width, bounding_rect.height) / 2;
			radius *=0.5;

			for( int i = 0; i < rows; ++i )
			{
				uchar* p_alpha = shadow_alpha.ptr<uchar>(i);
				
				for( int j = 0; j < cols; ++j )
				{
					float distance = sqrt( (double) ( j - down_point_x ) * ( j - down_point_x ) + ( i - down_point_y ) * ( i - down_point_y ) );

					if(distance < radius)
					{
						p_alpha[j] = 255;
					}
				}
			}

        }
    }

	bitwise_and(_alpha, shadow_alpha, shadow_alpha);
	morphologyEx(_alpha, _alpha, MORPH_CLOSE, element);

	Mat element2 = getStructuringElement( MORPH_ELLIPSE, cv::Size(15, 15), cv::Point(7, 7) );
	dilate(shadow_alpha, shadow_alpha, element);


//	cv::imshow("a", shadow_alpha);
//	cv::waitKey(0);

	return 0;
}

void do_transform(const cv::Matx33d &T, cv::vector<Point2f> &vpt)
{
	for(size_t i=0; i<vpt.size(); ++i)
	{
		cv::Matx31d tp(T*cv::Vec<double,3>(vpt[i].x, vpt[i].y, 1.0));
		vpt[i].x=tp(0)/tp(2);
		vpt[i].y=tp(1)/tp(2);
	}
}

int _draw_poly(Mat &draw, const vector<Point2f> &points, Scalar color, int thickness=1, int lineType=8)
{
	for(size_t i=0; i<points.size(); i++)
	{
		line(draw, points[i], points[(i+1)%points.size()], color, thickness, lineType);
	}

	return 0;
}

int _warp_back_corners(vector<Point2f> &corners, const Mat &T, Mat &img, const Scalar &color, int thickness=1, int lineType=8)
{
	do_transform(Matx33d(T), corners);

	_draw_poly(img, corners, color, thickness, lineType);
	return 0;
}

int _init_img_corners(const Mat &img, vector<Point2f> &corners)
{
	corners.clear();

	corners.push_back(Point2f(0, 0));
	corners.push_back(Point2f(img.cols, 0));
	corners.push_back(Point2f(img.cols, img.rows));
	corners.push_back(Point2f(0, img.rows));

	return 0;
}

void _translate_shape(TShape &shape, const Point2f &footPoint)
{
	Point2f center(shape.m_fg[0]+shape.m_fg[1]+shape.m_fg[2]);
	center.x/=3;
	center.y/=3;

	Point2f dv(footPoint-center);
	for(int i=0; i<3; ++i)
	{
		shape.m_bg[i]+=dv;
		shape.m_fg[i]+=dv;
	}
}

int _draw_triangle(Mat &draw, const Point2f points[], Scalar color, int thickness=1, int lineType=8)
{
//	assert(points.size() == 3);

	line(draw, points[0], points[1], color, thickness, lineType);
	line(draw, points[1], points[2], color, thickness, lineType);
	line(draw, points[2], points[0], color, thickness, lineType);

	return 0;
}

int _draw_triangle(Mat &draw, const Point2f points[], const Point2f &shift, Scalar color, int thickness=1, int lineType=8)
{
//	assert(points.size() == 3);

	line(draw, points[0]+shift, points[1]+shift, color, thickness, lineType);
	line(draw, points[1]+shift, points[2]+shift, color, thickness, lineType);
	line(draw, points[2]+shift, points[0]+shift, color, thickness, lineType);

	return 0;
}


namespace{

int _draw_points(Mat &img, Point2f points[], int count1, int count2, int radius, Scalar color1, Scalar color2, int thickness)
{
	for(int i=0; i<count1; i++)
	{
		cv::circle(img, points[i], 5, color1, thickness);
	}

	for(int i=count1; i<count1+count2; i++)
	{
		cv::circle(img, points[i], 5, color2, thickness);
	}

	return 0;
}

int _init_triangle(Point2f ref, float radius, vector<Point2f> &result)
{
	float ref_x = ref.x;
	float ref_y = ref.y;

	float x0 = ref_x;
	float y0 = ref_y - radius;

	float x1 = ref_x - 0.866 * radius;
	float y1 = ref_y + 0.5 * radius;

	float x2 = ref_x + 0.866 * radius;
	float y2 = ref_y + 0.5 * radius;

	result.clear();
	result.push_back(Point2f(x0, y0));
	result.push_back(Point2f(x1, y1));
	result.push_back(Point2f(x2, y2));

	return 0;
}

int _draw_triangle(Mat &draw, const vector<Point2f> &points, Scalar color, int thickness=1, int lineType=8)
{
	assert(points.size() == 3);

	line(draw, points[0], points[1], color, thickness, lineType);
	line(draw, points[1], points[2], color, thickness, lineType);
	line(draw, points[2], points[0], color, thickness, lineType);

	return 0;
}

Point2f _get_triangle_center(const vector<Point2f> &points)
{
	Point2f center = Point2f(0, 0);
	for(size_t i=0; i<points.size(); ++i)
	{
		center += points[i];
	}

	center.x /= points.size();
	center.y /= points.size();

	return center;
}

Point2f _transform_triangle(const Mat &H, vector<Point2f> &points, Point2f align_point)
{
	vector<Point2f> dst_points;
	perspectiveTransform(points, dst_points, H);

	Point2f center = _get_triangle_center(dst_points);
	Point2f bias = align_point - center;

	for(size_t i=0; i<dst_points.size(); ++i)
	{
		dst_points[i] += bias;
	}

	swap(points, dst_points);

	return bias;
}

int _transform_bg_triangle(const Mat &H, vector<Point2f> &points, Point2f foot_point, Point2f bias)
{
	Point2f center1 = _get_triangle_center(points);
	Point2f bias1 = foot_point - center1;

	for(size_t i=0; i<points.size(); ++i)
	{
		points[i] += bias1;
	}

	vector<Point2f> dst_points;
	perspectiveTransform(points, dst_points, H);

	for(size_t i=0; i<dst_points.size(); ++i)
	{
		dst_points[i] -= bias1;
	}

	for(size_t i=0; i<dst_points.size(); ++i)
	{
		dst_points[i] += bias;
	}

	swap(points, dst_points);

	return 0;
}

void _do_transform(Point2f pt[], const Point2f &shift, int n)
{
	for(int i=0; i<n; ++i)
	{
		pt[i].x=pt[i].x+shift.x;
		pt[i].y=pt[i].y+shift.y;
	}
}

void _init_cvt_points(cv::Point2f vpt[], float scale=50)
{
	vpt[0]=Point2f(-1,-1);
	vpt[1]=Point2f(-1, 1);
	vpt[2]=Point2f(1, 1);
	vpt[3]=Point2f(1, -1);

	vpt[4]=Point2f(-2.5,-2.5);
	vpt[5]=Point2f(-2.5, 2.5);
	vpt[6]=Point2f(2.5, 2.5);
	vpt[7]=Point2f(2.5, -2.5);


	for(int i=0; i<8; ++i)
		vpt[i]=Point2f(vpt[i].x*scale,vpt[i].y*scale);
}

void _set_identity(double d[], int m)
{
	for(int i=0; i<m; ++i)
	{
		for(int j=0; j<m; ++j)
		{
			d[m*i+j]=i==j? 1.0 : 0;
		}
	}
}

bool _inv_mat44(const double m[16], double d[16])
{
	const double a0=m[0], a1=m[1], a2=m[2], a3=m[3], a4=m[4], a5=m[5], a6=m[6], a7=m[7], a8=m[8], a9=m[9], a10=m[10], a11=m[11], a12=m[12], a13=m[13], a14=m[14], a15=m[15];

	double det=a0*a5*a10*a15 - a0*a5*a11*a14 - a0*a6*a9*a15 + a0*a6*a11*a13 + a0*a7*a9*a14 - a0*a7*a10*a13 - a1*a4*a10*a15 + a1*a4*a11*a14 + a1*a6*a8*a15 - a1*a6*a11*a12 - a1*a7*a8*a14 + a1*a7*a10*a12 + a2*a4*a9*a15 - a2*a4*a11*a13 - a2*a5*a8*a15 + a2*a5*a11*a12 + a2*a7*a8*a13 - a2*a7*a9*a12 - a3*a4*a9*a14 + a3*a4*a10*a13 + a3*a5*a8*a14 - a3*a5*a10*a12 - a3*a6*a8*a13 + a3*a6*a9*a12;

	if(fabs(det)<1e-8)
	{
//		_set_identity(d,4);
		return false;
	}

	double inv[16]=
	{
	a5*a10*a15 - a5*a11*a14 - a6*a9*a15 + a6*a11*a13 + a7*a9*a14 - a7*a10*a13, a1*a11*a14 - a1*a10*a15 + a2*a9*a15 - a2*a11*a13 - a3*a9*a14 + a3*a10*a13, a1*a6*a15 - a1*a7*a14 - a2*a5*a15 + a2*a7*a13 + a3*a5*a14 - a3*a6*a13, a1*a7*a10 - a1*a6*a11 + a2*a5*a11 - a2*a7*a9 - a3*a5*a10 + a3*a6*a9,
	a4*a11*a14 - a4*a10*a15 + a6*a8*a15 - a6*a11*a12 - a7*a8*a14 + a7*a10*a12, a0*a10*a15 - a0*a11*a14 - a2*a8*a15 + a2*a11*a12 + a3*a8*a14 - a3*a10*a12, a0*a7*a14 - a0*a6*a15 + a2*a4*a15 - a2*a7*a12 - a3*a4*a14 + a3*a6*a12, a0*a6*a11 - a0*a7*a10 - a2*a4*a11 + a2*a7*a8 + a3*a4*a10 - a3*a6*a8,
	a4*a9*a15 - a4*a11*a13 - a5*a8*a15 + a5*a11*a12 + a7*a8*a13 - a7*a9*a12,   a0*a11*a13 - a0*a9*a15 + a1*a8*a15 - a1*a11*a12 - a3*a8*a13 + a3*a9*a12, a0*a5*a15 - a0*a7*a13 - a1*a4*a15 + a1*a7*a12 + a3*a4*a13 - a3*a5*a12,   a0*a7*a9 - a0*a5*a11 + a1*a4*a11 - a1*a7*a8 - a3*a4*a9 + a3*a5*a8,
	a4*a10*a13 - a4*a9*a14 + a5*a8*a14 - a5*a10*a12 - a6*a8*a13 + a6*a9*a12,   a0*a9*a14 - a0*a10*a13 - a1*a8*a14 + a1*a10*a12 + a2*a8*a13 - a2*a9*a12, a0*a6*a13 - a0*a5*a14 + a1*a4*a14 - a1*a6*a12 - a2*a4*a13 + a2*a5*a12,   a0*a5*a10 - a0*a6*a9 - a1*a4*a10 + a1*a6*a8 + a2*a4*a9 - a2*a5*a8
	};

	for(int i=0; i<16; ++i)
		d[i]=inv[i]/det;

	return true;
}

bool _inv_mat33(const double m[9], double inv[9])
{
	const double a=m[0],b=m[1],c=m[2],d=m[3],e=m[4],f=m[5],g=m[6],h=m[7],i=m[8];

	double D=(-a*e*i+a*f*h+d*b*i-d*c*h-g*b*f+g*c*e);

	if(fabs(D)<1e-8)
	{
//		_set_identity(inv,3);
		return false;
	}

	D=1.0/D;
	inv[0]=-(e*i-f*h)*D, inv[1]=(b*i-c*h)*D, inv[2]=(-b*f+c*e)*D;
	inv[3]=(d*i-f*g)*D,  inv[4]=-(a*i-c*g)*D,inv[5]=-(-a*f+c*d)*D;
	inv[6]=-(d*h-e*g)*D, inv[7]=(a*h-b*g)*D, inv[8]= (-a*e+b*d)*D; 

	return true;
}

bool _inv_mat33( const cv::Matx33d &m, cv::Matx33d &inv)
{
	return _inv_mat33(m.val, inv.val);
}

template<int np>
bool get_tsr_matrix(const Point2f points1[], const Point2f points2[], const float wei[], cv::Matx33d &T)
{
	// 转换成解方程 A * X = B
	// X1 * S * cos - Y1 * sin + dx = X2;
	// X1 * sin + Y1 * S * cos + dy = Y2;

	enum{rows = 2 * np};
	cv::Matx<double, rows, 4> A;
	cv::Matx<double, rows, 1> B;

	for( unsigned int i=0; i<np; i++)
	{
		A(2*i, 0) = points1[i].x * wei[i];
		A(2*i, 1) = -points1[i].y * wei[i];
		A(2*i, 2) = wei[i];
		A(2*i, 3) = 0;

		A(2*i+1, 0) = points1[i].y * wei[i];
		A(2*i+1, 1) = points1[i].x * wei[i];
		A(2*i+1, 2) = 0;
		A(2*i+1, 3) = wei[i];

		B(2*i, 0) = points2[i].x * wei[i];
		B(2*i+1, 0) = points2[i].y * wei[i];
	}

//	solve( A, B, X, DECOMP_SVD ); 
	cv::Matx<double, 4, rows> AT(A.t());
	cv::Matx<double,4,1> BX(AT*B);

	cv::Matx<double,4,4> AAT(AT*A);
	if(!_inv_mat44(AAT.val, AAT.val))
		return false;

	BX=AAT*BX;

	double s_cos = BX(0);
	double s_sin = BX(1);
	double dx = BX(2);
	double dy = BX(3);
	//Mat tsr = (Mat_<double>(2, 3) << s_cos, -sin, dx, sin, s_cos, dy);
	T=cv::Matx33d(s_cos,-s_sin,dx,s_sin,s_cos,dy,0,0,1);

	return true;
}

template<int np>
bool get_ts_matrix(const Point2f points1[], const Point2f points2[], const float wei[], cv::Matx33d &T)
{
	// 转换成解方程 A * X = B
	// X1 * S * cos - Y1 * sin + dx = X2;
	// X1 * sin + Y1 * S * cos + dy = Y2;

	enum{rows = 2 * np};
	cv::Matx<double, rows, 3> A;
	cv::Matx<double, rows, 1> B;


	for( unsigned int i=0; i<np; i++)
	{
		A(2*i, 0) = points1[i].x * wei[i];
		A(2*i, 1) = wei[i];
		A(2*i, 2) = 0;

		A(2*i+1, 0) = points1[i].y * wei[i];
		A(2*i+1, 1) = 0;
		A(2*i+1, 2) = wei[i];

		B(2*i, 0) = points2[i].x * wei[i];
		B(2*i+1, 0) = points2[i].y * wei[i];
	}

	cv::Matx<double, 3, rows> AT(A.t());
	cv::Matx<double,3,1> BX(AT*B);

	cv::Matx<double,3,3> AAT(AT*A);
	if(!_inv_mat33(AAT.val, AAT.val))
		return false;

	BX=AAT*BX;

	double scale = BX(0);
	double dx = BX(1);
	double dy = BX(2);
	T=cv::Matx33d(scale,0,dx,0,scale,dy,0,0,1);

	return true;
}

template<int np>
bool get_tsr_matrix(const Point2f points1[], const Point2f points2[], cv::Matx33d &T)
{
	// 转换成解方程 A * X = B
	// X1 * S * cos - Y1 * sin + dx = X2;
	// X1 * sin + Y1 * S * cos + dy = Y2;

	enum{rows = 2 * np};
	cv::Matx<double, rows, 4> A;
	cv::Matx<double, rows, 1> B;

	for( unsigned int i=0; i<np; i++)
	{
		A(2*i, 0) = points1[i].x;
		A(2*i, 1) = -points1[i].y;
		A(2*i, 2) = 1;
		A(2*i, 3) = 0;

		A(2*i+1, 0) = points1[i].y;
		A(2*i+1, 1) = points1[i].x;
		A(2*i+1, 2) = 0;
		A(2*i+1, 3) = 1;

		B(2*i, 0) = points2[i].x;
		B(2*i+1, 0) = points2[i].y;
	}

//	solve( A, B, X, DECOMP_SVD ); 
	cv::Matx<double, 4, rows> AT(A.t());
	cv::Matx<double,4,1> BX(AT*B);

	cv::Matx<double,4,4> AAT(AT*A);
	if(!_inv_mat44(AAT.val, AAT.val))
		return false;

	BX=AAT*BX;

	double s_cos = BX(0);
	double s_sin = BX(1);
	double dx = BX(2);
	double dy = BX(3);
	//Mat tsr = (Mat_<double>(2, 3) << s_cos, -sin, dx, sin, s_cos, dy);
	T=cv::Matx33d(s_cos,-s_sin,dx,s_sin,s_cos,dy,0,0,1);

	return true;
}

template<int np>
bool get_ts_matrix(const Point2f points1[], const Point2f points2[], cv::Matx33d &T)
{
	// 转换成解方程 A * X = B
	// X1 * S * cos - Y1 * sin + dx = X2;
	// X1 * sin + Y1 * S * cos + dy = Y2;

	enum{rows = 2 * np};
	cv::Matx<double, rows, 3> A;
	cv::Matx<double, rows, 1> B;


	for( unsigned int i=0; i<np; i++)
	{
		A(2*i, 0) = points1[i].x;
		A(2*i, 1) = 1;
		A(2*i, 2) = 0;

		A(2*i+1, 0) = points1[i].y;
		A(2*i+1, 1) = 0;
		A(2*i+1, 2) = 1;

		B(2*i, 0) = points2[i].x;
		B(2*i+1, 0) = points2[i].y;
	}

	cv::Matx<double, 3, rows> AT(A.t());
	cv::Matx<double,3,1> BX(AT*B);

	cv::Matx<double,3,3> AAT(AT*A);
	if(!_inv_mat33(AAT.val, AAT.val))
		return false;

	BX=AAT*BX;

	double scale = BX(0);
	double dx = BX(1);
	double dy = BX(2);
	T=cv::Matx33d(scale,0,dx,0,scale,dy,0,0,1);

	return true;
}

template<int np>
bool _get_translation(const cv::Matx33d &T, cv::Matx33d &TX, cv::Point2f vpt[], float *error=NULL, bool rotation=false)
{
	cv::Point2f ptx[np];
	do_transform(T,vpt,ptx,np);

	bool ok=false;
	if(rotation)
		ok=get_tsr_matrix<np>(vpt,ptx,TX);
	else
		ok=get_ts_matrix<np>(vpt,ptx,TX);

	if(ok && error)
	{
		//cv::transform(pt1,ptx,TX);
		cv::Point2f dptx[np];
		do_transform(TX,vpt,dptx,np);
		*error=0;
		for(size_t i=0; i<np; ++i)
		{
			cv::Point2f dv(dptx[i]-ptx[i]);
			*error+=dv.dot(dv);
		}
		*error/=np;
	}

	return ok;
}

Scalar _interpolation_color(Scalar start_color, Scalar end_color, float ratio)
{
	int r0 = start_color[2];
	int g0 = start_color[1];
	int b0 = start_color[0];

	int r1 = end_color[2];
	int g1 = end_color[1];
	int b1 = end_color[0];

	int rn = (int)((r1 - r0 + 255) * ratio + r0) / 255;
	int gn = (int)((g1 - g0 + 255) * ratio + g0) / 255;
	int bn = (int)((b1 - b0 + 255) * ratio + b0) / 255;

	int re = (int)((r1 - r0 + 255) * ratio + r0) % 255;
	int ge = (int)((g1 - g0 + 255) * ratio + g0) % 255;
	int be = (int)((b1 - b0 + 255) * ratio + b0) % 255;

	int r = rn % 2 == 0? re : 255 - re;
	int g = gn % 2 == 0? ge : 255 - ge;
	int b = bn % 2 == 0? be : 255 - be;

	return Scalar(b, g, r);
}

}

void __blurRow(Mat &row, float blurRadius)
{

}

void __blurMask(Mat &mask, const Point2f &footPoint)
{
	int minRadius = 2;
	int maxRadius = 25;

	int footRow = footPoint.y;
	int rows = mask.rows;

	double blurRadius = 0;
	for(int i=rows-1; i>=0; --i)
	{
		
		if(i>=footRow)
		{
			blurRadius = minRadius;
		}
		else
		{
			float distanceRatio = (float)i / (float)footRow;
			blurRadius = minRadius + (maxRadius - minRadius) * (1 - distanceRatio);
		}
	
		Mat curRow = mask.row(i);
		blur(curRow, curRow, Size(blurRadius, blurRadius), cv::Point(-1, -1), BORDER_REPLICATE);
		//__blurRow(curRow, blurRadius);
	}

}

int CSlippage::Output(DPParam param, const vector< vector<Mat> > &v_shadow_H, float level, bool show_foot_points )
{
	
	FgDataSet fgDataSet;
	BgDataSet bgDataSet;

	if(this->_LoadDataSet(&fgDataSet, &bgDataSet)<0)
		return -1;

	fgDataSet.SetStartEndPos(param.foreground_start_end[0], param.foreground_start_end[1]);

	ofstream outputFile("fig.txt");

	VideoWriter src_video, bg_video, new_video, new_large_video, sel_video, path_video;

	for(uint i=0; i<fgDataSet.Size(); ++i)
	{
		std::cout << "frame:" << i <<endl;
		FgFrameData *fg_fd = fgDataSet.GetAtIndex(i);

		int correspond_bg_index = fg_fd->m_Correspondence;
		Mat correspond_bg_affine = fg_fd->m_CorrespondenceAffine;

		std::cout << correspond_bg_affine << endl;

		int videoId = bgDataSet.GetAtIndex(correspond_bg_index)->m_videoID;
		int frameId = bgDataSet.GetAtIndex(correspond_bg_index)->m_FrameID;



//===============================output txt file to matlab===============
	/*	static int preVideoId = -1;
		static int preFrameId = -1;

		if(i==0)
		{
			outputFile << i << "\t" << 0 << "\t" << videoId << "\n";
		}
		else
		{
			int step = (videoId == preVideoId? (frameId - preFrameId) : 0);
			outputFile << i << "\t" << step << "\t" << videoId << "\n";
		}

		preVideoId = videoId;
		preFrameId = frameId;

		continue;	*/
//======================================================================

		ImageSetReader&bgIsr(*bgIsrList[videoId]);

		fgIsrPtr->SetPos(i+param.foreground_start_end[0]);
		bgIsr.SetPos(frameId);

		Mat curAlpha;
		for(size_t k=0; k<alphaIsrList.size(); ++k)
		{
			ImageSetReader &alphaIsr(*(alphaIsrList[k]));
			Mat tempAlpha = alphaIsr.Read(i+param.foreground_start_end[0])->clone();
			cv::cvtColor(tempAlpha,tempAlpha,CV_BGR2GRAY);

			if(k==0)
				curAlpha = tempAlpha.clone();
			else
				bitwise_or(curAlpha, tempAlpha, curAlpha);
		}
		
		Mat *fg_img = fgIsrPtr->Read();
		Mat *fg_alpha = &curAlpha;
		Mat *bg_img = bgIsr.Read();
		Mat bg_img_warp;

		cv::resize(*fg_img, *fg_img, Size(), m_fgWorkScale, m_fgWorkScale);
		cv::resize(*fg_alpha, *fg_alpha, Size(), m_fgWorkScale, m_fgWorkScale);

		/////////////////////////////extension foreground/////////////////////////////////
		int extension_up = param.foreground_extension[0];
		int extension_down = param.foreground_extension[1];
		int extension_left = param.foreground_extension[2];
		int extension_right = param.foreground_extension[3];

		Mat extension_transform = ( Mat_<double>(3,3) << 1, 0, extension_left, 0, 1, extension_up, 0, 0, 1 );
		Mat extension_transform_inv = ( Mat_<double>(3,3) << 1, 0, -extension_left, 0, 1, -extension_up, 0, 0, 1 );
		correspond_bg_affine = extension_transform * correspond_bg_affine;

		int result_height = fg_img->rows + extension_up + extension_down;
		int result_width = fg_img->cols + extension_left + extension_right;


		cv::warpPerspective(*fg_img, *fg_img, extension_transform, Size(result_width, result_height));
		cv::warpPerspective(*fg_alpha, *fg_alpha, extension_transform, Size(result_width, result_height));


		/////////////////////////////////warp//////////////////////////////

		if(correspond_bg_affine.rows==2)
			warpAffine(*bg_img, bg_img_warp, correspond_bg_affine, Size(result_width, result_height));
		else
			cv::warpPerspective(*bg_img, bg_img_warp, correspond_bg_affine, Size(result_width, result_height));

		if(i == 0)
			bg_video = VideoWriter(dataDir+"result_bg.avi", CV_FOURCC('D','I','V','X'), 25, Size(fg_img->cols, fg_img->rows));

		bg_video.write(bg_img_warp);

		Mat fg_imgx=fg_img->clone();

		////////////////////////////////////add shadow/////////////////////////////////////////

		if(v_shadow_H.size() > 0)
		{
			Mat fg_alpha_warp_all;

			for(int k=0; k<alphaIsrList.size(); ++k)
			{
				ImageSetReader &alphaIsr(*(alphaIsrList[k]));
				Mat tempAlpha = alphaIsr.Read(i+param.foreground_start_end[0])->clone();
				cv::cvtColor(tempAlpha,tempAlpha,CV_BGR2GRAY);
				cv::resize(tempAlpha, tempAlpha, Size(), m_fgWorkScale, m_fgWorkScale);

				Mat shadow_H = extension_transform * v_shadow_H[i][k];
				cv::warpPerspective(tempAlpha, tempAlpha, shadow_H, fg_alpha->size());

				if(k==0)
				{
					fg_alpha_warp_all = tempAlpha.clone();
				}
				else
				{
					bitwise_or(tempAlpha, fg_alpha_warp_all, fg_alpha_warp_all);
				}
			}

			_synthesis_shadow(bg_img_warp, fg_alpha_warp_all, level);
		}

		/////////////////////////////////////////////////////////////////////////////

		//warp back the foreground region to the selected background image
		Mat bg_imgx = (*bg_img).clone();
		vector<Point2f> fg_corners;
		_init_img_corners(fg_imgx, fg_corners);
		_warp_back_corners(fg_corners, correspond_bg_affine.inv(), bg_imgx, Scalar(0, 255, 0), 5, 8);


		Mat new_img, new_large_img;
		synthesis_new_img(fg_imgx, *fg_alpha, bg_img_warp, bg_imgx, 0, 0, i, videoId, frameId, new_img, new_large_img, fg_fd->m_TransformMethod, true);

		if( i == 0 )
		{
			sel_video = VideoWriter(dataDir+"result_sel.avi", CV_FOURCC('D','I','V','X'), 25, Size(new_img.cols, new_img.rows));
			src_video = VideoWriter(dataDir+"result_src.avi", CV_FOURCC('D','I','V','X'), 25, Size(fg_imgx.cols, fg_imgx.rows));
			new_video = VideoWriter(dataDir+"result.avi", CV_FOURCC('D','I','V','X'), 25, Size(new_img.cols, new_img.rows));
			new_large_video = VideoWriter(dataDir+"result_large.avi", CV_FOURCC('D','I','V','X'), 25, Size(new_large_img.cols, new_large_img.rows));
		}

		if(show_foot_points)
		{
		//	cv::circle(new_img, cvPoint(fg_fd->m_FootPoints.front().x + extension_left, fg_fd->m_FootPoints.front().y + extension_up),9, cvScalar(45,255,250),-1);

			Point2f _footPoints[8];
			_init_cvt_points(_footPoints);
			_do_transform(_footPoints, fg_fd->m_FootPoints.front(), 8);

		//	_draw_points(new_img, _footPoints, 4, 4, 5, Scalar(85, 176, 0), Scalar(255, 255, 0), -1);
		}

		resize(bg_imgx, bg_imgx, fg_imgx.size());
		sel_video.write(bg_imgx);
		src_video.write(fg_imgx);
		new_video.write(new_img);
		new_large_video.write(new_large_img);

		if(new_large_img.cols>1920)
			cv::resize(new_large_img,new_large_img, cv::Size(1920,new_large_img.rows*1920/new_large_img.cols));

		cv::imshow("new video", new_large_img);
	//	cv::imshow("new_img", new_img);
		cv::waitKey(1);
	}

	outputFile.close();
	path_video.release();
	sel_video.release();
	src_video.release();
	bg_video.release();
	new_video.release();
	new_large_video.release();
	return 0;
	

}

int CSlippage::OutputUnclip(DPParam param, const vector< vector<Mat> > &v_shadow_H, float level, bool show_foot_points )
{
	FgDataSet fgDataSet;
	BgDataSet bgDataSet;

	if(this->_LoadDataSet(&fgDataSet, &bgDataSet)<0)
		return -1;

	ff::OBFStream os(ff::CatDirectory(this->GetDataDir(),"unclip.db"), true);
	deque<UnclipedData> vUnclipedData;

	fgDataSet.SetStartEndPos(param.foreground_start_end[0], param.foreground_start_end[1]);

	VideoWriter src_video, bg_video, new_video, new_large_video, sel_video, path_video;

	for(uint i=0; i<fgDataSet.Size(); ++i)
	{
		std::cout << "frame:" << i <<endl;
		FgFrameData *fg_fd = fgDataSet.GetAtIndex(i);

		int correspond_bg_index = fg_fd->m_Correspondence;
		Mat correspond_bg_affine = fg_fd->m_CorrespondenceAffine;

		std::cout << correspond_bg_affine << endl;

		int videoId = bgDataSet.GetAtIndex(correspond_bg_index)->m_videoID;
		int frameId = bgDataSet.GetAtIndex(correspond_bg_index)->m_FrameID;
		ImageSetReader&bgIsr(*bgIsrList[videoId]);

		fgIsrPtr->SetPos(i+param.foreground_start_end[0]);
		bgIsr.SetPos(frameId);

		Mat curAlpha;
		for(size_t k=0; k<alphaIsrList.size(); ++k)
		{
			ImageSetReader &alphaIsr(*(alphaIsrList[k]));
			Mat tempAlpha = alphaIsr.Read(i+param.foreground_start_end[0])->clone();
			cv::cvtColor(tempAlpha,tempAlpha,CV_BGR2GRAY);

			if(k==0)
				curAlpha = tempAlpha.clone();
			else
				bitwise_or(curAlpha, tempAlpha, curAlpha);
		}
		
		Mat *fg_img = fgIsrPtr->Read();
		Mat *fg_alpha = &curAlpha;
		Mat *bg_img = bgIsr.Read();
		Mat bg_img_warp;

		cv::resize(*fg_img, *fg_img, Size(), m_fgWorkScale, m_fgWorkScale);
		cv::resize(*fg_alpha, *fg_alpha, Size(), m_fgWorkScale, m_fgWorkScale);

		/////////////////////////////extension foreground/////////////////////////////////
		int extension_up = param.foreground_extension[0];
		int extension_down = param.foreground_extension[1];
		int extension_left = param.foreground_extension[2];
		int extension_right = param.foreground_extension[3];

		Mat extension_transform = ( Mat_<double>(3,3) << 1, 0, extension_left, 0, 1, extension_up, 0, 0, 1 );
		Mat extension_transform_inv = ( Mat_<double>(3,3) << 1, 0, -extension_left, 0, 1, -extension_up, 0, 0, 1 );
		correspond_bg_affine = extension_transform * correspond_bg_affine;

		int result_height = fg_img->rows + extension_up + extension_down;
		int result_width = fg_img->cols + extension_left + extension_right;


		cv::warpPerspective(*fg_img, *fg_img, extension_transform, Size(result_width, result_height));
		cv::warpPerspective(*fg_alpha, *fg_alpha, extension_transform, Size(result_width, result_height));


		/////////////////////////////////warp//////////////////////////////

#define UNCLIP
#ifdef UNCLIP
		cv::Point2f  bgCorners[]={Point2f(0,0),Point2f(0,bg_img->rows),Point2f(bg_img->cols,bg_img->rows),Point2f(bg_img->cols,0)}, bgCornersWarped[4];
		affine_to_homogeneous(correspond_bg_affine);
		do_transform(correspond_bg_affine,bgCorners,Point2f(0,0),bgCornersWarped,4);
		std::vector<cv::Point2f>  vcorners(bgCornersWarped, bgCornersWarped+4);
		cv::Rect bb(_get_bounding_box(vcorners));

		Mat bg_img_warp_uncliped;
		Point tl = bb.tl();

	//	Mat alignH = (Mat_<double>(3, 3) << 1, 0, -tl.x, 0, 1, -tl.y, 0, 0, 1);
	//	Mat correspond_bg_affine_unCliped = alignH * correspond_bg_affine ;
	//	cv::warpPerspective(*bg_img, bg_img_warp_uncliped, correspond_bg_affine_unCliped, Size(bb.width, bb.height));

		Mat alignH = (Mat_<double>(3, 3) << 1, 0, 20, 0, 1, 20, 0, 0, 1);
		Mat correspond_bg_affine_unCliped = alignH * correspond_bg_affine;
		cv::warpPerspective(*bg_img, bg_img_warp_uncliped, correspond_bg_affine_unCliped, Size(result_width+40, result_height+40));

		//write the unclip data
		UnclipedData ud;
		ud.fgClip = Rect(-tl.x, -tl.y, result_width, result_height);
		for(int m = 0; m<4; ++m)
		{
			int x = bgCornersWarped[m].x - tl.x;
			int y = bgCornersWarped[m].y - tl.y;
			ud.validPixelCorners.push_back(Point(x, y));
		}
		vUnclipedData.push_back(ud);

#endif

		if(correspond_bg_affine.rows==2)
			warpAffine(*bg_img, bg_img_warp, correspond_bg_affine, Size(result_width, result_height));
		else
			cv::warpPerspective(*bg_img, bg_img_warp, correspond_bg_affine, Size(result_width, result_height));

		if(i == 0)
			bg_video = VideoWriter(dataDir+"result_bg.avi", CV_FOURCC('D','I','V','X'), 25, Size(fg_img->cols, fg_img->rows));

		bg_video.write(bg_img_warp);

		Mat fg_imgx=fg_img->clone();

		////////////////////////////////////add shadow/////////////////////////////////////////

		if(v_shadow_H.size() > 0)
		{
			Mat fg_alpha_warp_all;

			for(int k=0; k<alphaIsrList.size(); ++k)
			{
				ImageSetReader &alphaIsr(*(alphaIsrList[k]));
				Mat tempAlpha = alphaIsr.Read(i+param.foreground_start_end[0])->clone();
				cv::cvtColor(tempAlpha,tempAlpha,CV_BGR2GRAY);
				cv::resize(tempAlpha, tempAlpha, Size(), m_fgWorkScale, m_fgWorkScale);

				//Point2f footPoint = fg_fd->m_FootPoints[k];
				//__blurMask(tempAlpha, footPoint);
				//imshow("aa", tempAlpha);
				//waitKey(0);
				
				Mat shadow_H = extension_transform * v_shadow_H[i][k];
				cv::warpPerspective(tempAlpha, tempAlpha, shadow_H, fg_alpha->size());

				if(k==0)
				{
					fg_alpha_warp_all = tempAlpha.clone();
				}
				else
				{
					bitwise_or(tempAlpha, fg_alpha_warp_all, fg_alpha_warp_all);
				}
			}

			_synthesis_shadow(bg_img_warp, fg_alpha_warp_all, level);

#ifdef UNCLIP

			warpPerspective(fg_alpha_warp_all, fg_alpha_warp_all, alignH, bg_img_warp_uncliped.size());
			_synthesis_shadow(bg_img_warp_uncliped, fg_alpha_warp_all, level);
			
			Mat fg_alpha_uncliped, fg_img_uncliped;
			warpPerspective(fg_imgx, fg_img_uncliped, alignH, bg_img_warp_uncliped.size());
			warpPerspective(*fg_alpha, fg_alpha_uncliped, alignH, bg_img_warp_uncliped.size());
			
			Mat newImgUncliped;
			synthesis_new_img(fg_img_uncliped, fg_alpha_uncliped, bg_img_warp_uncliped, newImgUncliped);

			stringstream fName, fName_bg;
			fName << "B_uncliped_result\\" << setw(3) << setfill('0') << i << ".png";
			imwrite(ff::CatDirectory(this->GetDataDir(), fName.str()), newImgUncliped);
			imshow("newImgUncliped", newImgUncliped);

			fName_bg << "B_uncliped\\" << setw(3)  << setfill('0') << i << ".png";
			imwrite(ff::CatDirectory(this->GetDataDir(), fName_bg.str()), bg_img_warp_uncliped);
			imshow("bg_img_warp_uncliped", bg_img_warp_uncliped);			
#endif
		}

		/////////////////////////////////////////////////////////////////////////////

		//warp back the foreground region to the selected background image
		Mat bg_imgx = (*bg_img).clone();
		vector<Point2f> fg_corners;
		_init_img_corners(fg_imgx, fg_corners);
		_warp_back_corners(fg_corners, correspond_bg_affine.inv(), bg_imgx, Scalar(0, 255, 0), 5, 8);


		Mat new_img, new_large_img;
		synthesis_new_img(fg_imgx, *fg_alpha, bg_img_warp, bg_imgx, 0, 0, i, videoId, frameId, new_img, new_large_img, fg_fd->m_TransformMethod, true);

		Mat path_img = new_img.clone();

		TShape shapex(g_shape[i]);
		_translate_shape(shapex, fg_fd->m_FootPoints.front());
		_draw_triangle(path_img, shapex.m_fg, Scalar(0, 0, 255), 2, CV_AA );
		_draw_triangle(path_img, shapex.m_bg, Scalar(0, 255, 0), 2, CV_AA );


		if( i == 0 )
		{
			path_video = VideoWriter(dataDir+"result_path.avi", CV_FOURCC('D','I','V','X'), 25, new_img.size());;
			sel_video = VideoWriter(dataDir+"result_sel.avi", CV_FOURCC('D','I','V','X'), 25, Size(new_img.cols, new_img.rows));
			src_video = VideoWriter(dataDir+"result_src.avi", CV_FOURCC('D','I','V','X'), 25, Size(fg_imgx.cols, fg_imgx.rows));
			new_video = VideoWriter(dataDir+"result.avi", CV_FOURCC('D','I','V','X'), 25, Size(new_img.cols, new_img.rows));
			new_large_video = VideoWriter(dataDir+"result_large.avi", CV_FOURCC('D','I','V','X'), 25, Size(new_large_img.cols, new_large_img.rows));
		}

		if(show_foot_points)
		{
		//	cv::circle(new_img, cvPoint(fg_fd->m_FootPoints.front().x + extension_left, fg_fd->m_FootPoints.front().y + extension_up),9, cvScalar(45,255,250),-1);

			Point2f _footPoints[8];
			_init_cvt_points(_footPoints);
			_do_transform(_footPoints, fg_fd->m_FootPoints.front(), 8);

		//	_draw_points(new_img, _footPoints, 4, 4, 5, Scalar(85, 176, 0), Scalar(255, 255, 0), -1);
		}

		path_video.write(path_img);
		resize(bg_imgx, bg_imgx, fg_imgx.size());
		sel_video.write(bg_imgx);
		src_video.write(fg_imgx);
		new_video.write(new_img);
		new_large_video.write(new_large_img);

		if(new_large_img.cols>1920)
			cv::resize(new_large_img,new_large_img, cv::Size(1920,new_large_img.rows*1920/new_large_img.cols));

		cv::imshow("new video", new_large_img);
	//	cv::imshow("new_img", new_img);
		cv::waitKey(1);
	}

	path_video.release();
	sel_video.release();
	src_video.release();
	bg_video.release();
	new_video.release();
	new_large_video.release();

#ifdef UNCLIP
	os << vUnclipedData;
//	cout << vUnclipedData.size() << endl;
//	ff::IBFStream osread(ff::CatDirectory(this->GetDataDir(),"unclip.db"));
//	vUnclipedData.clear();
//	osread >> vUnclipedData;
	cout << vUnclipedData.size();
#endif

	return 0;
}

int CSlippage::ShowSelectedFrame(DPParam param, SSFParam ssf_param)
{
	int video_count = bgIsrList.size();

	int video_width = ssf_param.width;
	int video_height = ssf_param.height * video_count + ssf_param.interval_height * (video_count - 1);

	Mat bgImg = Mat(video_height, video_width, CV_8UC3, ssf_param.bg_color);

	for(int i=0; i<video_count; ++i)
	{
		Mat cur_video_img = Mat(ssf_param.height, ssf_param.width, CV_8UC3, ssf_param.colors[i]);
		Mat cur_roi = bgImg(Rect(0, i*ssf_param.height+i*ssf_param.interval_height, ssf_param.width, ssf_param.height));
		cur_video_img.copyTo(cur_roi);
	}

	FgDataSet fgDataSet;
	BgDataSet bgDataSet;

	if(this->_LoadDataSet(&fgDataSet, &bgDataSet)<0)
		return -1;

	fgDataSet.SetStartEndPos(param.foreground_start_end[0], param.foreground_start_end[1]);

	VideoWriter new_video = VideoWriter(dataDir+"result_show_selected_frames.avi", CV_FOURCC('D','I','V','X'), 25, Size(video_width, video_height));

	for(uint i=0; i<fgDataSet.Size(); ++i)
	{
		std::cout << "frame:" << i <<endl;

		FgFrameData *fg_fd = fgDataSet.GetAtIndex(i);
		int correspond_bg_index = fg_fd->m_Correspondence;

		int videoId = bgDataSet.GetAtIndex(correspond_bg_index)->m_videoID;
		int frameId = bgDataSet.GetAtIndex(correspond_bg_index)->m_FrameID;

		Mat new_img = bgImg.clone();

		int dx, dy, w, h;

		int cur_video_size = bgIsrList[videoId]->Size();

		dx = ((float)frameId / (float)cur_video_size) * video_width;
		dy = videoId * (ssf_param.height + ssf_param.interval_height);
		w = ssf_param.selected_width;
		h = ssf_param.height;

		if(videoId == 1)
		{
			dx = video_width - dx;
		}

		if(dx + w > new_img.cols)
		{
			w = new_img.cols - dx;
		}
		if(dy + h > new_img.rows)
		{
			h = new_img.rows - dy;
		}

		Mat cur_roi = new_img(Rect(0, dy, new_img.cols, h));
		cur_roi = ssf_param.selected_bar_color;

		cur_roi = new_img(Rect(dx, dy, w, h));
		cur_roi = ssf_param.selected_color;

		new_video.write(new_img);

		cv::imshow("new video", new_img);
		cv::waitKey(1);
	}

	new_video.release();

	return 0;
}

int CSlippage::ShowSelectedFrame2(DPParam param, SSFParam2 ssf_param)
{
	FgDataSet fgDataSet;
	BgDataSet bgDataSet;

	if(this->_LoadDataSet(&fgDataSet, &bgDataSet)<0)
		return -1;

	fgDataSet.SetStartEndPos(param.foreground_start_end[0], param.foreground_start_end[1]);


	Mat drawImg = Mat::zeros(ssf_param.height, fgDataSet.Size(), CV_8UC3);

	int start_bg = INT_MAX;
	int end_bg = -1;
	for(size_t i=0; i<fgDataSet.Size(); ++i)
	{
		FgFrameData *fg_fd = fgDataSet.GetAtIndex(i);
		int correspond_bg_index = fg_fd->m_Correspondence;
		if(correspond_bg_index > end_bg)	end_bg = correspond_bg_index;
		if(correspond_bg_index < start_bg)  start_bg = correspond_bg_index;
	}
	
	for(size_t i=0; i<fgDataSet.Size(); ++i)
	{
		std::cout << "frame:" << i <<endl;

		FgFrameData *fg_fd = fgDataSet.GetAtIndex(i);
		int correspond_bg_index = fg_fd->m_Correspondence;

		float ratio = (float)(correspond_bg_index - start_bg) / (float)(end_bg - start_bg);
		Scalar color = _interpolation_color(ssf_param.start_color, ssf_param.end_color, ratio);

		Mat drawImg_roi = drawImg(Rect(i, 0, 1, ssf_param.height));
		drawImg_roi = color;

		cout << ratio << endl << correspond_bg_index << endl;
		imshow("a", drawImg);
		waitKey(1);
	}

	imwrite(dataDir+"fg_bg_img.jpg", drawImg);
	waitKey(1);

	return 0;
}