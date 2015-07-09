
#ifndef _FF_IFF_IMAGE_H
#define _FF_IFF_IMAGE_H


#include "bfc\err.h"

#include "bfc\type.h"
#include "bfc\stdf.h"
#include "bfc\bfstream.h"
#include<vector>

#include "iff\def.h"

_IFF_BEG

const int FI_LOCK_DATA=0x0100;

class _IFF_API FVTImage
{
protected:
	uchar*   _pData;
	int	    _width,_height,_step;
	int     _type;
	int	    _flag;
public:
	FVTImage()
		:_pData(0),_width(0),_height(0),_step(0),_type(FI_8UC1),_flag(0)
	{
	}
	FVTImage(void* pData,int width,int height,int type,int step)
		:_pData((uchar*)pData),_width(width),_height(height),_type(type),_step(step),_flag(0)
	{
	}
	FVTImage(int width,int height,int type,int align=4);

	FVTImage(const FVTImage& right);

	~FVTImage() throw();

	FVTImage& operator=(const FVTImage& right);

	void Lock(int mask=FI_LOCK_DATA);

	void Unlock(int mask=FI_LOCK_DATA);

	bool DataLocked() const
	{
		return (_flag&FI_LOCK_DATA)!=0;
	}
	void Clear();

	void Swap(FVTImage& right);
	
	//reshape or reformat the image data.
	//@newWidth<=0 : caculate the width according to data size per row!
	void Reshape(int newType, int newWidth =0 );

	void Reset(int width,int height,int type,int align=4);

	void StepReset(int width,int height,int type,int step);

	void ResetIf(int width,int height,int type);

	void Attach(void* pData,int width,int height,int type,int step);

	bool IsExternData() const;
	
	const uchar* Data() const
	{
		return _pData;
	}
	const uchar* Data(int x,int y) const
	{
		return _pData+y*_step+x*this->PixelSize();
	}
	uchar* Data()
	{
		return _pData;
	}
	uchar* Data(int x,int y)
	{
		return _pData+y*_step+x*this->PixelSize();
	}
	template<typename _Ty>
	_Ty*  DataAs()
	{
		return (_Ty*)_pData;
	}
	template<typename _Ty>
	_Ty* DataAs(int x,int y) 
	{
		return (_Ty*)this->Data(x,y);
	}
	template<typename _Ty>
	const _Ty*  DataAs() const
	{
		return (_Ty*)_pData;
	}
	template<typename _Ty>
	const _Ty* DataAs(int x,int y) const
	{
		return (_Ty*)this->Data(x,y);
	}
	int   Width() const
	{
		return _width;
	}
	int   Height() const
	{
		return _height;
	}
	int  Step() const
	{
		return  _step;
	}
	int  Stride() const
	{
		assert(_step%FI_DEPTH_SIZE(_type)==0);

		return _step/FI_DEPTH_SIZE(_type);
	}
	int  LineSize() const
	{
		return this->PixelSize()*this->Width();
	}
	int  Type() const
	{
		return  _type;
	}
	int  NChannels() const
	{
		return  FI_CN(_type);
	}
	int  Depth()  const
	{
		return  FI_DEPTH(_type);
	}
	int  PixelSize() const
	{
		return  FI_TYPE_SIZE(_type);
	}
	bool IsNull() const
	{
		return this->Data()==0||this->Width()<=0||this->Height()<=0;
	}
	Rect GetRect() const
	{
		return Rect(0,0,this->Width(),this->Height());
	}
	void AttachROI(const FVTImage& img,const Rect& roi);

};

template<typename _BufferT>
inline void BFSWrite(OBStream<_BufferT> &os,const FVTImage &img)
{
	os<<img.Width()<<img.Height()<<img.Type()<<img.Step();
	const uchar *pData=img.Data();

	for(int yi=0;yi<img.Height();++yi,pData+=img.Step())
		os.Write(pData,img.LineSize(),1);
}

template<typename _BufferT>
inline void BFSRead(IBStream<_BufferT> &is, FVTImage &img)
{
	int width=is.Read<int>(),height=is.Read<int>(),type=is.Read<int>(),step=is.Read<int>();
	img.StepReset(width,height,type,step);
	
	uchar *pData=img.Data();
	for(int yi=0;yi<height;++yi,pData+=img.Step())
		is.Read(pData,img.LineSize(),1);
}

//******************************************************************************************************

enum
{
	FI_VAPT_NO_INIT=0x01,
	FI_VAPT_ZERO_INIT=0x02,
	FI_VAPT_BIT_COPY=0x04,
	FI_VAPT_NO_DESTRUCT=0x08
};

template<typename _PixelValT,int _PixelTraits=0>
class ValArray2D
{
	enum{_VAF_INIT_MASK=0x03,_VAF_EXDATA=0x01};

	typedef ValArray2D<_PixelValT,_PixelTraits> _MyT;
public:
	typedef _PixelValT  PixelValueType;
protected:
	_PixelValT *m_pData;
	int			m_width,m_height,m_step;
	int			m_flag;
private:
	_PixelValT* _alloc_data(int width,int height)
	{
		if(width<=0||height<=0)
			return NULL;

		_PixelValT *pData=NULL;
		int step=width*sizeof(_PixelValT);

		pData=(_PixelValT*)::operator new(step*height);

		if((_PixelTraits&_VAF_INIT_MASK)==FI_VAPT_ZERO_INIT)
			memset(pData,0,step*height);
		else
			if((_PixelTraits&_VAF_INIT_MASK)==FI_VAPT_NO_INIT)
			{//do nothing.
			}
			else
			{//default construct.
				int np=width*height;
				for(int i=0;i<np;++i)
					::new (&pData[i]) _PixelValT();
			}

		return pData;
	}
	void _release_data(_PixelValT *pData,int width,int height)
	{
		if(!(_PixelTraits&FI_VAPT_NO_DESTRUCT))
		{
			int np=width*height; 
			for(int i=0;i<np;++i)
				pData[i].~_PixelValT();
		}
		::operator delete(pData);
	}
	void _clear(bool bReinit)
	{
		if(!this->IsExternData())
			this->_release_data(m_pData,m_width,m_height);
		if(bReinit)
		{
			m_pData=NULL;
			m_width=m_height=m_step=m_flag=0;
		}
	}
public:
	static void CopyData(const _PixelValT *pSrc,int width,int height,int istep,
							_PixelValT  *pDest,int ostep
							)
	{
		if(_PixelTraits&FI_VAPT_BIT_COPY)
		{
			if(istep==ostep)
				memcpy(pDest,pSrc,istep*height);
			else
			{
				for(int yi=0;yi<height;++yi,pSrc=ByteDiff(pSrc,istep),pDest=ByteDiff(pDest,ostep))
					memcpy(pDest,pSrc,width*sizeof(_PixelValT));
			}
		}
		else
		{
			for(int yi=0;yi<height;++yi,pSrc=ByteDiff(pSrc,istep),pDest=ByteDiff(pDest,ostep))
			{
				for(int xi=0;xi<width;++xi)
					pDest[xi]=pSrc[xi];
			}
		}
	}
public:
	ValArray2D()
		:m_pData(NULL)
	{
		m_width=m_height=m_step=m_flag=0;
	}
	ValArray2D(int width,int height)
	{
		m_pData=_alloc_data(width,height);
		m_step=sizeof(_PixelValT)*width;
		m_width=width;
		m_height=height;
		m_flag=0;
	}
	ValArray2D(_PixelValT *pData,int width,int height,int step)
		:m_pData(NULL),m_flag(_VAF_EXDATA)
	{
		this->Attach(pData,width,height,step);
	}
	ValArray2D(const ValArray2D &right)
	{
		m_pData=_alloc_data(right.Width(),right.Height());
		m_step=sizeof(_PixelValT)*right.Width();
		m_width=right.Width();
		m_height=right.Height();
		m_flag=0;

		CopyData(right.m_pData,right.Width(),right.Height(),right.Step(),m_pData,m_step);
	}
	_MyT& operator=(const _MyT &right)
	{
		if(this!=&right)
		{
			_MyT temp(right);
			this->Swap(temp);
		}
		return *this;
	}
	~ValArray2D() throw()
	{
		this->_clear(false);
	}
	void Swap(ValArray2D &right)
	{
		SwapObjectMemory(*this,right);
	}

	bool IsExternData() const
	{
		return (m_flag&_VAF_EXDATA)!=0;
	}
	void Clear()
	{
		this->_clear(true);
	}
	void Reset(int width,int height)
	{
		_MyT temp(width,height);
		this->Swap(temp);
	}
	void ResetIf(int width,int height)
	{
		if(width!=m_width||height!=m_height)
			this->Reset(width,height);
	}
	void Attach(_PixelValT *pData,int width,int height,int step)
	{
		this->_clear(false);
		m_pData=pData;
		m_width=width,m_height=height,m_step=step;
		m_flag|=_VAF_EXDATA;
	}
	int Width() const
	{
		return m_width;
	}
	int Height() const
	{
		return m_height;
	}
	int Step() const
	{
		return m_step;
	}
	int Stride() const
	{
		assert(m_step%sizeof(_PixelValT)==0);

		return m_step/sizeof(_PixelValT);
	}
	int PixelSize() const
	{
		return sizeof(_PixelValT);
	}
	int LineSize() const
	{
		return sizeof(_PixelValT)*m_width;
	}
	bool IsNull() const
	{
		return m_pData==NULL;
	}
	Rect GetRect() const
	{
		return Rect(0,0,m_width,m_height);
	}
	const _PixelValT * Data() const
	{
		return m_pData;
	}
	_PixelValT * Data()
	{
		return m_pData;
	}
	const _PixelValT * Data(int x,int y) const
	{
		return ByteDiff(m_pData,y*m_step)+x;
	}
	_PixelValT * Data(int x,int y)
	{
		return ByteDiff(m_pData,y*m_step)+x;
	}
	template<typename _DestT>
	_DestT * DataAs()
	{
		return (_DestT*)m_pData;
	}
	template<typename _DestT>
	const _DestT * DataAs(int x,int y) const
	{
		return (_DestT*)(ByteDiff(m_pData,y*m_step)+x);
	}
	template<typename _DestT>
	_DestT * Data(int x,int y)
	{
		return (_DestT*)(ByteDiff(m_pData,y*m_step)+x);
	}
	void AttachROI(const _MyT &img,const Rect &roi)
	{
		if(this!=&img)
		{
			Rect roix(OverlappedRect(roi,img.Rect()));

			if(roix.IsEmpty())
				this->Clear(); //clear if ROI is empty.           
			else
				this->Attach((_PixelValT*)img.Data(roix.X(),roix.Y()),roix.Width(),roix.Height(),img.Step());
		}
		else
		{
			//attach part of the data to the image itself, can't finished.
			if(roi!=img.Rect())
				FF_EXCEPT(ERR_INVALID_OP,"");
		}
	}
public:

	template<typename _BufT>
	friend void BFSWrite(OBStream<_BufT> &os,const _MyT &arr)
	{
		os<<arr.Width()<<arr.Height();
		
		const PixelValueType *pData=arr.Data();
		for(int yi=0;yi<arr.Height();++yi,pData=ByteDiff(pData,arr.Step()))
		{
			const PixelValueType *pdx=pData;
			for(int xi=0;xi<arr.Width();++xi,++pdx)
				os<<*pdx;
		}
	}
	template<typename _BufT>
	friend void BFSRead(IBStream<_BufT> &is, _MyT &arr)
	{
		int width=is.Read<int>(),height=is.Read<int>();

		arr.ResetIf(width,height);
		PixelValueType *pData=arr.Data();
		for(int yi=0;yi<height;++yi,pData=ByteDiff(pData,arr.Step()))
		{
			PixelValueType *pdx=pData;
			for(int xi=0;xi<width;++xi,++pdx)
				is>>*pdx;
		}
	}
};


class ImageArray
{
	std::vector<FVTImage*>  m_vImg;
public:
	ImageArray()
	{
	}
	ImageArray(const FVTImage *vImg, int count=1)
	{
		for(int i=0; i<count; ++i)
		{
			this->Append(vImg+i);
		}
	}
	void Append(const FVTImage *img)
	{
		m_vImg.push_back((FVTImage*)img);
	}

	void Clear()
	{
		m_vImg.clear();
	}

	int  Size() const
	{
		return (int)m_vImg.size();
	}

	const FVTImage& operator[](int i) const
	{
		return *m_vImg[i];
	}
	FVTImage& operator[](int i)
	{
		return *m_vImg[i];
	}
};


//******************************************************************************************************

template<typename _FIT0,typename _FIT1>
inline bool fiSizeEq( _FIT0 & img0, _FIT1 &img1)
{
	return img0.Width()==img1.Width()&&img0.Height()==img1.Height();
}

//////////////////////////////////////////////

#define FI_AL_WHS(img)  (img).Width(),(img).Height(),(img).Step()
//#define FI_AL_WHST(img)  FI_AL_WHS(img),(img).Type()
//#define FI_AL_WHSC(img)  FI_AL_WHS(img),(img).NChannels()
//#define FI_AL_WHSP(img)  FI_AL_WHS(img),(img).PixelSize()
//
//#define FI_AL_ST(img)  (img).Step(),(img).Type()
//#define FI_AL_SC(img)  (img).Step(),(img).NChannels()
//#define FI_AL_SP(img)  (img).Step(),(img).PixelSize()

#define FI_AL_DWHS_AS(img,type) (img).DataAs<type>(),FI_AL_WHS(img)
#define _DWHS_AS FI_AL_DWHS_AS

#define FI_AL_DWHS(img)	  (img).Data(),FI_AL_WHS(img)
#define _DWHS FI_AL_DWHS

//#define FI_AL_DWHST_AS(img,type) FI_AL_DWHS_AS(img,type),(img).Type()
//#define FI_AL_DWHST(img)  FI_AL_DWHS(img),(img).Type()

#define FI_AL_DWHSC_AS(img,type) FI_AL_DWHS_AS(img,type),(img).NChannels()
#define _DWHSC_AS FI_AL_DWHSC_AS

#define FI_AL_DWHSC(img)  FI_AL_DWHS(img),(img).NChannels()
#define _DWHSC FI_AL_DWHSC

#define FI_AL_DWHSP_AS(img,type) FI_AL_DWHS_AS(img,type),(img).PixelSize()
#define _DWHSP_AS FI_AL_DWHSP_AS

#define FI_AL_DWHSP(img)  FI_AL_DWHS(img),(img).PixelSize()
#define _DWHSP FI_AL_DWHSP

#define FI_AL_DS_AS(img,type)   (img).DataAs<type>(),(img).Step()
#define _DS_AS FI_AL_DS_AS

#define FI_AL_DS(img)	  (img).Data(),(img).Step()
#define _DS FI_AL_DS
//
//#define FI_AL_DST_AS(img,type) FI_AL_DS_AS(img,type),(img).Type()
//#define FI_AL_DST(img)  FI_AL_DS(img),(img).Type()
//
#define FI_AL_DSC_AS(img,type) FI_AL_DS_AS(img,type),(img).NChannels()
#define _DSC_AS FI_AL_DSC_AS

#define FI_AL_DSC(img)  FI_AL_DS(img),(img).NChannels()
#define _DSC FI_AL_DSC

#define FI_AL_DSP_AS(img,type) FI_AL_DS_AS(img,type),(img).PixelSize()
#define _DSP_AS FI_AL_DSP_AS

#define FI_AL_DSP(img)  FI_AL_DS(img),(img).PixelSize()
#define _DSP FI_AL_DSP



#define FI_AL_WHN(img)  (img).Width(),(img).Height(),(img).Stride()

#define FI_AL_DWHN_AS(img,type) (img).DataAs<type>(),FI_AL_WHN(img)
#define FI_AL_DWHN(img)	  (img).Data(),FI_AL_WHN(img)

#define FI_AL_DWHNC_AS(img,type) FI_AL_DWHN_AS(img,type),(img).NChannels()
#define FI_AL_DWHNC(img)  FI_AL_DWHN(img),(img).NChannels()

#define FI_AL_DN_AS(img,type)   (img).DataAs<type>(),(img).Stride()
#define FI_AL_DN(img)	  (img).Data(),(img).Stride()

#define FI_AL_DNC_AS(img,type) FI_AL_DN_AS(img,type),(img).NChannels()
#define FI_AL_DNC(img)  FI_AL_DN(img),(img).NChannels()

_IFF_END

#endif

