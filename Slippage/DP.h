
#ifndef _INC_DP_H
#define _INC_DP_H
//
#include<vector>
#include"opencv2/core/core.hpp"

//
//typedef ff::SmallMatrix<double,3,3> Matrix33d;
//
//struct _FgFrameData
//{
//public:
//	Matrix33d		m_RelativeAffine;
//	ff::Point2f		m_FootPoint;
//};

struct _FgFrameData
{
public:
	cv::Matx33d		m_RelativeAffine;
	cv::Point2f		m_FootPoint;
};

struct _BgNbr
{
public:
	int		m_index;
//	double  m_T[3][3];
	cv::Matx33d  m_T;
};

struct _BgFrameData
{
public:
	int			m_npt;
	cv::Point2f  *m_vpt;

	int			m_nnbr;
	_BgNbr	m_vnbr[1];
public:
	cv::Point2f* GetPolyPoints()
	{
		return m_vpt;
	}
};


enum{N_FOOT_NBRS=5};

inline void   init_cvt_points(cv::Point2f vpt[], float scale=50)
{
	vpt[0]=Point2f(0,0);
	vpt[1]=Point2f(-1,-1);
	vpt[2]=Point2f(-1, 1);
	vpt[3]=Point2f(1, 1);
	vpt[4]=Point2f(1, -1);

	//vpt[4]=Point2f(0,-1);
	//vpt[5]=Point2f(-1,0);
	//vpt[6]=Point2f(0,1);
	//vpt[7]=Point2f(1,0);

	for(int i=1; i<N_FOOT_NBRS; ++i)
		vpt[i]=Point2f(vpt[i].x*scale,vpt[i].y*scale);
}

inline void do_transform(const cv::Matx33d &T, const Point2f pt[], const Point2f &shift, Point2f dpt[], int n)
{
	for(int i=0; i<n; ++i)
	{
		cv::Matx31d vi(pt[i].x+shift.x, pt[i].y+shift.y, 1.0);

		vi=T*vi;
		dpt[i].x=vi(0)/vi(2);
		dpt[i].y=vi(1)/vi(2);
	}
}

inline void do_transform(const cv::Matx33d &T, const cv::Point2f vpt[], cv::Point2f dpt[], int np)
{
	for(int i=0; i<np; ++i)
	{
		cv::Matx31d tp(T*cv::Vec<double,3>(vpt[i].x, vpt[i].y, 1.0));
		dpt[i].x=tp(0)/tp(2);
		dpt[i].y=tp(1)/tp(2);
	}
}

template<typename _PointT>
inline cv::Rect_<float> _get_bounding_box(const std::vector<_PointT> &vpt, float MAX_SIZE=1e5)
{
	float l,t,r,b;
	l=r=vpt[0].x;
	t=b=vpt[0].y;

	for(size_t i=0; i<vpt.size(); ++i)
	{
		if(fabs(vpt[i].x)<MAX_SIZE)
		{
			if(vpt[i].x<l) 
				l=vpt[i].x;
			else if(vpt[i].x>r) r=vpt[i].x;
		}

		if(fabs(vpt[i].y)<MAX_SIZE)
		{
			if(vpt[i].y<t)
				t=vpt[i].y;
			else if(vpt[i].y>b) b=vpt[i].y;
		}
	}

//	return cv::Rect_<float>(int(l+0.5),int(t+0.5),int(r-l+0.5),int(b-t+0.5));
	return cv::Rect_<float>(l,t,r-l,b-t);
}



inline double _get_bg_distortion(const cv::Point2f dpt[4])
{
	//must the four corners arranged counter clockwise
	float derr=fabs(dpt[0].x-dpt[1].x)+fabs(dpt[2].x-dpt[3].x)+fabs(dpt[0].y-dpt[3].y)+fabs(dpt[1].y-dpt[2].y);
	derr/=4;

	Point2f dv1( dpt[2]-dpt[0] ), dv2(dpt[3]-dpt[1]);
	float L=(sqrt(dv1.dot(dv1))+sqrt(dv2.dot(dv2)))/2;

	return derr*100/L; //do normalize
}


inline double _poly_area(const cv::Point2f vpt[4])
{
	double s=0;

	for(size_t i=1; i<4; ++i)
	{
		s+=(vpt[i-1].x-vpt[i].x)*(vpt[i-1].y+vpt[i].y);
	}
	s+=(vpt[3].x-vpt[0].x)*(vpt[3].y+vpt[0].y);
	
	return fabs(s*0.5);
}

inline double _poly_area(const cv::Point2f vpt[], int count)
{
	double s=0;

	for(size_t i=1; i<count; ++i)
	{
		s+=(vpt[i-1].x-vpt[i].x)*(vpt[i-1].y+vpt[i].y);
	}
	s+=(vpt[count-1].x-vpt[0].x)*(vpt[count-1].y+vpt[0].y);
	
	return fabs(s*0.5);
}


inline double _tri_area(const Point2f &p0, const Point2f &p1, const Point2f &p2)
{
	double s=(p0.x-p1.x)*(p0.y+p1.y);
	s+=(p1.x-p2.x)*(p1.y+p2.y);
	s+=(p2.x-p0.x)*(p2.y+p0.y);

	return fabs(s*0.5);
}

inline double _get_coverage(const cv::Point2f bpt[4], const cv::Point2f fpt[4], double fgArea)
{
	double t_error=0;
	double bg_quad_area=_poly_area(bpt);

	if(bg_quad_area<fgArea*0.5)
		swap(bpt,fpt);

	for(size_t i=0; i<4; ++i)
	{
		double pt_area=_tri_area(fpt[i],bpt[0],bpt[1])+_tri_area(fpt[i],bpt[1],bpt[2])+_tri_area(fpt[i],bpt[2],bpt[3])+_tri_area(fpt[i],bpt[3],bpt[0]);
		t_error+=fabs(pt_area-bg_quad_area);
	}
	//t_error*=0.25;

	return (float)t_error;
}

inline double _get_coverage(const cv::Point2f bpt[], int nbpt, double fgArea)
{
	double bgArea=_poly_area(bpt, nbpt);

	return fabs(fgArea-bgArea)/fgArea;
}


template<typename _PointT>
inline float _get_bounding_area(const std::vector<_PointT> &vpt)
{
	cv::Rect_<float> bb(_get_bounding_box(vpt));

	return bb.width*bb.height;
}


template<typename _PointT>
inline float _get_distortion(const std::vector<_PointT> &dpt)
{
	float AREA = _get_bounding_area(dpt);
	float area1 = _poly_area(dpt);
//	float area0 = _poly_area(pt);
//	double scale=__min(area0,area1)/__max(area0,area1);

	return 1-area1/AREA;
}

template<typename _PointT>
inline float _get_transform_distortion(const cv::Mat &T, const std::vector<_PointT> &pt)
{
	std::vector<cv::Point3f>  dpt;
	cv::transform(pt,dpt,T);

	for(size_t i=0; i<pt.size(); ++i)
	{
		if(fabs(dpt[i].z)<1e-6f)
			return 1.0f;

		dpt[i].x/=dpt[i].z;
		dpt[i].y/=dpt[i].z;
	}

	return _get_distortion(dpt);
}

struct TShape
{
	Point2f m_fg[3];
	Point2f m_bg[3];
};


#endif

