
#include"stdafx.h"

#include"imgset.h"

#include<algorithm>

ImageSetReader::ImageSetReader()
	:m_id(-1), m_dsize(0,0)
{
}

ImageSetReader::~ImageSetReader()
{
}

bool ImageSetReader::ReadSized(int pos, cv::Mat &img, const cv::Size &dsize)
{
	cv::Mat *_img=this->Read(pos);
	if(!_img)
		return false;

	if(dsize.width==0||dsize.height==0)
		img=_img->clone();
	else
		cv::resize(*_img,img,dsize,0,0,cv::INTER_NEAREST);

	return true;
}

bool ImageSetReader::ReadSized(int pos, cv::Mat &img)
{
	return this->ReadSized(pos,img,m_dsize);
}

//==================================================================================

#if 0
#include"capx/capx.h"

ISRVideo::ISRVideo()
:m_pCap(NULL),m_pCurImg(new Mat)
{
	m_width=m_height=0;
}

ISRVideo::~ISRVideo()
{
	delete m_pCurImg;
	capxReleaseCapture(m_pCap);
}

int  ISRVideo::Create(const string_t &file)
{
	int ec=-1;

	static bool s_comInit=false;

	if(!s_comInit)
	{
		CoInitialize(NULL);
		s_comInit=true;
	}

	capxCapture *pcap=capxCreateVideo(ff::charset::_2MBS(file).c_str());

	if(pcap)
	{
		capxSetFlag(pcap, CAPX_FLIP);

		if(m_pCap)
			capxReleaseCapture(m_pCap);

		m_pCap=pcap;

		capxOutputProperty op;
		capxGetOutputProperty(m_pCap,&op);

		m_width=op.width;
		m_height=op.height;
		ec=0;
	}

	return ec;
}

int ISRVideo::Size()
{
	return capxGetNumberOfFrames(m_pCap);
}

int ISRVideo::GetUFID(int pos)
{
	return make_ufid(this->GetID(), this->Pos());
}

int ISRVideo::Width()
{
	return m_width;
}

int ISRVideo::Height()
{
	return m_height;
}

Mat* ISRVideo::Read()
{
	capxOutputProperty op;
	void *pdata=capxGrabFrame(m_pCap,&op);
	if(pdata)
	{
	//	m_pCurImg->Attach(pdata,op.width,op.height,FI_MAKE_TYPE(FI_8U,CAPX_PIXEL_SIZE(op.format)),op.step);

		cv::Mat img(op.height,op.width, CV_MAKETYPE(CV_8U, CAPX_PIXEL_SIZE(op.format)), pdata, op.step);

		assert(m_pCurImg);
		cv::swap(img, *m_pCurImg);
		
		return m_pCurImg;
	}
	return NULL;
}

bool ISRVideo::MoveForward()
{
	capxState ec=capxSeek(m_pCap,1,CAPX_SEEK_RELATIVE);
	return ec>0;
}

int ISRVideo::Pos()
{
	return capxGetCurrentPosition(m_pCap);
}

int ISRVideo::SetPos(int pos)
{
	capxState ec=capxSeek(m_pCap,pos);
	
	return ec<0? ec : 0;
}

string_t ISRVideo::FrameName(int pos)
{
	char_t buf[16];
	return _itot(pos,buf,10);
}
#endif

//=====================================================================================

ISRImages::ISRImages()
{
	m_width=m_height=0;
	m_bufPos=m_pos=-1;
}

uint _get_file_id(WIN32_FIND_DATA *di)
{
	const char_t *pe=di->cFileName+_tcslen(di->cFileName);
	const char_t *px=di->cFileName;

	while(px!=pe && !_istdigit(*px) ) ++px;

	uint id=uint(-1);

	if(px!=pe)
	{
		const char_t *pb=px;
		while(px!=pe && _istdigit(*px) ) ++px;
		string_t sid(pb,px);
		id=_ttoi(sid.c_str());
	}

	return id;
}


template<typename _PairT>
struct less_first
{
public:
	bool operator()(const _PairT &left, const _PairT &right)
	{
		return left.first<right.first;
	}
};

template<typename _PairT>
struct less_second
{
public:
	bool operator()(const _PairT &left, const _PairT &right)
	{
		return left.second<right.second;
	}
};

int ListFiles(const string_t &path, const string_t &ext, std::vector<ISRImages::FilePairT> &flist)
{
	WIN32_FIND_DATA ffd;
	HANDLE hFind=::FindFirstFile((path+_T("\\*.")+ext).c_str(),&ffd);

	if(hFind==INVALID_HANDLE_VALUE)
		return -1;

	typedef std::pair<string_t,uint>  _PairT;
	std::vector< _PairT>  vlist;

	bool compare_by_id=true;
	do
	{
		string_t extention(ff::GetFileExtention(ffd.cFileName));
		if(extention=="jpg"||extention=="png"||extention=="bmp")
		{
			uint id=0;
		
			if(compare_by_id)
			{
				id=_get_file_id(&ffd);
				if(id==uint(-1)) //if failed
					compare_by_id=false;
			}

			vlist.push_back(_PairT(ffd.cFileName,id) );
		}
	}
	while(::FindNextFile(hFind,&ffd));

	::FindClose(hFind);

	if(compare_by_id)
		std::sort(vlist.begin(), vlist.end(), less_second<_PairT>());
	else
	//	std::sort(vlist.begin(), vlist.end(), less_first<_PairT>());
		return -1;

//	for(size_t i=0; i<vlist.size(); ++i)
//		flist.push_back(vlist[i].first);
	
	flist.swap(vlist);

	return 0;
}

int ISRImages::Create(const string_t &file, int nc)
{
	string_t dir(ff::GetDirectory(file)), ext(ff::GetFileExtention(file));
	ff::RevisePath(dir,ff::RP_FULL_PATH);

	int ec=ListFiles(dir,ext,m_vfiles);

	if(ec!=0)
		return ec;

	//std::sort(m_vfiles.begin(), m_vfiles.end());
	m_dir=dir+_T("\\");
	m_nc=nc;

	ec=-1;

	if(this->Size()>0)
	{
		const Mat *first=NULL;

		if(this->SetPos(0)>=0 && (first=this->Read()))
		{
			m_width=first->cols;
			m_height=first->rows;
			ec=0;
		}
	}
	else
		m_width=m_height=0;

	return ec;
}

int ISRImages::GetFileIndex(const string_t &name)
{
	for(size_t i=0; i<m_vfiles.size(); ++i)
	{
		if(_tcsnicmp(name.c_str(),m_vfiles[i].first.c_str(),name.size())==0)
			return (int)i;
	}

	return -1;
}

int ISRImages::Size()
{
	return (int)m_vfiles.size();
}

int ISRImages::GetUFID(int pos)
{
	return  uint(pos)<m_vfiles.size()? make_ufid(this->GetID(), m_vfiles[pos].second) : -1;
}

int ISRImages::Width()
{
	return m_width;
}

int ISRImages::Height()
{
	return m_height;
}

Mat* ISRImages::Read()
{
	if(m_bufPos!=m_pos)
	{
		bool ok=false;

		if(uint(m_pos)<uint(m_vfiles.size()))
		{
			try
			{
				Mat T(cv::imread(m_dir+m_vfiles[m_pos].first));

				//convert number of channels if necessary
				if(m_nc>0 && T.channels()!=m_nc)
				{
				//	fiConvertRGBChannels(T,m_bufImg,m_nc);
					bool number_of_channels_mismatch=false;
					assert(number_of_channels_mismatch);
				}
				else
					cv::swap(T,m_bufImg);

				m_bufPos=m_pos;
				ok=true;
			}
			catch(...)
			{}
		}

		if(!ok)
		{
			m_bufPos=-1;
		}
	}

	return m_bufPos==m_pos && uint(m_pos)<uint(m_vfiles.size()) ? &m_bufImg : NULL;
	
}

bool ISRImages::MoveForward()
{
	return this->SetPos(m_pos+1)==0? true:false;
}

int ISRImages::Pos()
{
	return m_pos;
}

int ISRImages::SetPos(int pos)
{
	if(uint(pos)<uint(this->Size()))
	{
		m_pos=pos;
		return 0;
	}
	return -1;
}

string_t ISRImages::FrameName(int pos)
{
	if(uint(pos)<uint(this->Size()))
		return m_vfiles[pos].first;

	static string_t nullName;

	return nullName;
}

