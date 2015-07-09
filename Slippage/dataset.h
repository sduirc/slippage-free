
#pragma once

#include"BFC\stdf.h"
#include"BFC\bfstream.h"
#include"BFC\autores.h"
#include"BFC\diskmap.h"
#include<vector>
#include<deque>
#include<map>

using namespace std;
using namespace cv;

namespace cv
{
	template<typename _IBST>
	inline void BSRead(_IBST &ibs, Mat &m) 
	{ 
		int width=ibs.Read<int>(),height=ibs.Read<int>(),depth=ibs.Read<int>(),cn=ibs.Read<int>();
		m.create(height,width,CV_MAKETYPE(depth,cn));
		for(int yi=0; yi<height; ++yi)
		{
			ibs.Read(m.ptr(yi),m.elemSize()*width,1);
		}
	} 

	template<typename _OBST>
	inline void BSWrite(_OBST &obs,const Mat &m) 
	{
		obs<<m.cols<<m.rows<<m.depth()<<m.channels();

		for(int yi=0; yi<m.rows; ++yi)
		{
			obs.Write(m.ptr(yi),m.elemSize()*m.cols,1);
		}
	}

	template<typename _IBST, typename _ValT>
	inline void BSRead(_IBST &ibs, cv::Point_<_ValT> &m) 
	{ 
		ibs.Read(&m,sizeof(m),1);
	} 

	template<typename _OBST, typename _ValT>
	inline void BSWrite(_OBST &obs,const cv::Point_<_ValT> &m) 
	{
		obs.Write(&m,sizeof(m),1);
	}

	template<typename _IBST>
	inline void BSRead(_IBST &ibs, KeyPoint &m) 
	{ 
		ibs.Read(&m,sizeof(m),1);
	} 

	template<typename _OBST>
	inline void BSWrite(_OBST &obs,const KeyPoint &m) 
	{
		obs.Write(&m,sizeof(m),1);
	}

	template<typename _IBST>
	inline void BSRead(_IBST &ibs, Rect &m) 
	{ 
		ibs.Read(&m,sizeof(m),1);
	} 

	template<typename _OBST>
	inline void BSWrite(_OBST &obs,const Rect &m) 
	{
		obs.Write(&m,sizeof(m),1);
	}

	template<typename _IBST>
	inline void BSRead(_IBST &ibs, Point &m) 
	{ 
		ibs.Read(&m,sizeof(m),1);
	} 

	template<typename _OBST>
	inline void BSWrite(_OBST &obs,const Point &m) 
	{
		obs.Write(&m,sizeof(m),1);
	}

}

enum
{
	MOTION_TS,
	MOTION_TSR,
	MOTION_HOMOGRAPHY
};

struct DPError
{
	float  m_coverage;
	float  m_bgDistortion;
	float  m_matching;
	float  m_smooth;
public:
	DEFINE_BFS_IO_4(DPError,m_coverage,m_bgDistortion,m_matching,m_smooth)
};

struct FgFrameData
{
public:
	Mat				m_RelativeAffine;
	int				m_Correspondence;
	int				m_TransformMethod;
	Mat				m_CorrespondenceAffine;	
	Mat             m_CorrespondenceHomography;	
	vector<Point2f> m_FootPoints;
	DPError         m_Errors;
public:
	DEFINE_BFS_IO_7(FgFrameData,m_RelativeAffine,m_Correspondence,m_TransformMethod,m_CorrespondenceAffine,m_CorrespondenceHomography,m_FootPoints, m_Errors)
};

/* Data of each frame

*/
struct  BgFrameData
{
	struct MyMatch
	{
		ushort idx1;
		ushort idx2;

	public :
		MyMatch(ushort _idx1, ushort _idx2):idx1(_idx1),idx2(_idx2)
		{
		}
		MyMatch()
		{
		}
	public:
		DEFINE_BFS_IO_2(MyMatch, idx1, idx2)
	};

	enum
	{
		NF_STITCH_CAND=0x01
	};

	struct Neighbor
	{
		int		m_index;
		Mat		m_T;
		float	m_distortion_err;
		float	m_transform_err;
		std::vector<MyMatch> m_nbr_match;
		int     m_flag;
		std::vector<cv::Mat>  m_subH;
	public:
		Neighbor()
			:m_flag(0)
		{
		}

	//	DEFINE_BFS_IO_6(Neighbor,m_index,m_T,m_distortion_err,m_transform_err,m_nbr_match)
		template<typename _IBST>
		friend void BSRead(_IBST &ibs,Neighbor &v) 
		{ 
			ibs >> v.m_index >> v.m_T >> v.m_distortion_err >>v.m_transform_err >> v.m_nbr_match >>v.m_flag >> v.m_subH;
		} 
		template<typename _OBST>
		friend void BSWrite(_OBST &obs,const Neighbor &v) 
		{ 
			obs << v.m_index << v.m_T << v.m_distortion_err <<v.m_transform_err << v.m_nbr_match << v.m_flag << v.m_subH;
		} 
	};
public:
	// all
	vector<Point2f>			m_keyPoints;

	std::vector<Neighbor>	m_nbr;
	int						m_nTemporalNBR;

	vector<cv::Point2f>       m_polyRegion;

	int				m_videoID;
	int				m_FrameID;

public:
	template<typename _IBST>
	friend void BSRead(_IBST &ibs,BgFrameData &v) 
	{ 
		ibs >> v.m_keyPoints>>v.m_nbr>>v.m_nTemporalNBR>>v.m_videoID>>v.m_FrameID>>v.m_polyRegion;
	} 
	template<typename _OBST>
	friend void BSWrite(_OBST &obs,const BgFrameData &v) 
	{ 
		obs << v.m_keyPoints<<v.m_nbr<<v.m_nTemporalNBR<<v.m_videoID<<v.m_FrameID<<v.m_polyRegion;
	} 
};

/* Manage data of frames

_DataT : data type of each frame
_KeyT  : a unique key to index frames

*/
template<typename _DataT, typename _KeyT>
class DataSet
{
public:
	typedef _DataT  DataType;
	typedef _KeyT	KeyType;
private:
	typedef  int  _IndexT;
	typedef std::pair<_KeyT,_IndexT>	_KeyPairT;
	typedef std::map<_KeyT,_IndexT> _KeyMapT;
	
	_KeyMapT	m_keyMap;
	
	std::deque<_DataT>	m_dataList;

	int m_startPos;
	int m_endPos;

private:
	_IndexT  _GetIndex(const _KeyT &key)
	{
		_KeyMapT::iterator itr(m_keyMap.find(key));

		return itr==m_keyMap.end()? -1 : itr->second;
	}

	_IndexT  _NewData(const _KeyT &key)
	{
		_IndexT index=(_IndexT)m_dataList.size();
		std::pair<_KeyMapT::iterator,bool> ib(m_keyMap.insert(_KeyPairT(key,index)));

		if(ib.second)
		{
			m_dataList.push_back(_DataT());
			return index;
		}
		return -1;
	}

public:

	DataSet()
	{
		m_startPos = 0;
		m_endPos   = -1;   // not include itself
	}

	int SetStartEndPos(int start, int end)
	{
		if(start < 0)
			return -1;

		this->m_startPos = start;
		this->m_endPos = end;

		return 0;
	}


	_IndexT  GetIndex(const _KeyT &key)
	{
		_KeyMapT::iterator itr(m_keyMap.find(key));

		return itr==m_keyMap.end()? -1 : itr->second;
	}

	//Get exist data, return NULL if not exist
	_DataT* GetAt(const _KeyT &key)
	{
		_IndexT index=this->_GetIndex(key);

		return size_t(index)<m_dataList.size()? &m_dataList[index] : NULL;
	}

	//Get or create data : return the data if exist, otherwise create it.
	_DataT* Get(const _KeyT &key)
	{
		_IndexT index=this->_GetIndex(key);

		if(size_t(index)>=m_dataList.size())
		{
			index=this->_NewData(key); 
		}
		assert(size_t(index)<m_dataList.size());

		return &m_dataList[index];
	}

	_DataT* GetAtIndex(const _IndexT &index)
	{
		int endIndex = m_endPos < 0? m_dataList.size() : m_endPos;
		int size = endIndex - m_startPos;

		return size_t(index)<size_t(size)? &m_dataList[index+m_startPos] : NULL;
	}


	uint Size()
	{
		int endIndex = m_endPos < 0? m_dataList.size() : m_endPos;
		int size = endIndex - m_startPos;

		return size;
	//	return 100;
	//	return m_dataList.size()>300? 100 : m_dataList.size();
	}

	//save the dataset, index and data are saved separately as a .idx and  a .db file
	//@_file : can either the index file (.idx) or the database file(.db)
	int Save(const string_t &_file)
	{
		int ec=-1;

		try
		{
			{//save index file
				string_t file(ff::ReplacePathElem(_file,"idx",ff::RPE_FILE_EXTENTION));
				ff::OBFStream os(file,true);
			
				std::vector<_KeyPairT>  vkey;
				vkey.reserve(m_keyMap.size());
				for(_KeyMapT::const_iterator itr(m_keyMap.begin()); itr!=m_keyMap.end(); ++itr)
				{
					vkey.push_back(*itr);
				}

				os<<vkey;
			}

			{//save data file
				string_t file(ff::ReplacePathElem(_file,"db",ff::RPE_FILE_EXTENTION));
				ff::OBFStream os(file,true);

				os<<m_dataList;
			}

			ec=0;
		}
		catch(...)
		{}

		return ec;
	}

	int Load(const string_t &_file)
	{
		int ec=-1;

		try
		{
			{
				string_t file(ff::ReplacePathElem(_file,"idx",ff::RPE_FILE_EXTENTION));
				if(ff::IsExistPath(file))
				{
					ff::IBFStream is(file);

					std::vector<_KeyPairT> vkey;
					is>>vkey;

					m_keyMap.clear();
					for(size_t i=0; i<vkey.size(); ++i)
					{
						m_keyMap.insert(vkey[i]);
					}
				}
			}

			{
				string_t file(ff::ReplacePathElem(_file,"db",ff::RPE_FILE_EXTENTION));
				if(ff::IsExistPath(file))
				{
					ff::IBFStream is(file);

					is>>m_dataList;
				}
			}

			ec=0;
		}
		catch(...)
		{}

		return ec;
	}
};


typedef DataSet<FgFrameData,int>  FgDataSet;
typedef DataSet<BgFrameData,int>  BgDataSet;

struct BgFrameDataEx
{
	cv::Mat  m_descriptor;
	std::vector<cv::Point2f>  m_KLTFeatures;
public:
	DEFINE_BFS_IO_2(BgFrameDataEx,m_descriptor, m_KLTFeatures)
};


typedef ff::DiskMap<BgFrameDataEx, ff::DMIndexMap<int,int>, int, int>  BgDiskMap;


struct SlippageData
{
	cv::Mat   m_InitBT;
	double    m_bgWorkScale;
public:
	DEFINE_BFS_IO_2(SlippageData, m_InitBT, m_bgWorkScale)
};

struct UnclipedData
{
	cv::Rect fgClip;
	vector<Point_<int>> validPixelCorners;

public:
	DEFINE_BFS_IO_2(UnclipedData, fgClip, validPixelCorners)
};