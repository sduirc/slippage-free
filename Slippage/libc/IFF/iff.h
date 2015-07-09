
#ifndef _FF_IFF_IFF_H
#define _FF_IFF_IFF_H

#include"iff\image.h"
#include"iff\util.h"
#include"iff/itf.h"


_IFF_BEG


////////////////////////////////////////////////////////
//memory allocation used by class @FVTImage.

_IFF_API void* fiAlloc(uint size);

_IFF_API uchar* fiAllocImageData(const int step,const int height);

_IFF_API uchar* fiAllocImageData(const int width,const int height,const int type,const int align=4);

//free memory allocated with the above three functions.
_IFF_API void   fiFree(void* ptr);


//caculate step accroding to line size and alignment.
//@lineSize = width*ps, where ps is the pixel size in bytes.
inline int fiStep(const int lineSize,const int align)
{
	return lineSize%align==0? lineSize:(lineSize/align+1)*align;
}
//caculate step according to image width, pixel type and alignment.
inline int fiStep(const int width,const int type,const int align)
{
	return fiStep(width*FI_PIXEL_SIZE(type),align);
}
//get the alignment according to step.
//Note from step to alignment is ambiguity, this function return only the maximum alignment that
//fit the specified step.
inline int fiAlign(const int step)
{
	return step%8==0? 8:step%4==0? 4:1;
}

const int FI_COPY_FLIP=0x01;
const int FI_COPY_SWAP_RB=0x02;

void   _IFF_API fiCopy(const FVTImage& src,FVTImage& dest,const int mode=0);

void   _IFF_API fiSetMem(FVTImage& img,const char val);

void  _IFF_API  fiSetBoundary(FVTImage &img,const int bw,const char val);

void   _IFF_API fiSetPixel(FVTImage& img,const void* pPixelVal);

void   _IFF_API fiSetBoundaryPixel(FVTImage &img,const int bw,const void *pPixelVal);


const int FI_AXIS_HORZ=0;
const int FI_AXIS_VERT=1;

void _IFF_API fiFlip(const FVTImage& src,FVTImage& dest,const int axis=FI_AXIS_HORZ);

void _IFF_API fiFlip(FVTImage& srcDest,const int axis=FI_AXIS_HORZ);

void _IFF_API fiTile(const FVTImage& src,FVTImage& dest,const int destWidth,const int destHeight);

//dest=src*@scale+@shift.
void _IFF_API fiTransformPixel(const FVTImage& src,FVTImage& dest,const int destDepth,
							const double scale=1.0,const double shift=0.0);

//dest=src;
void _IFF_API fiCastDepth(const FVTImage& src,FVTImage& dest,const int destDepth);
//swap two channels.
//@ic0,@ic1, zero based index of the channels to be swapped.
void _IFF_API fiSwapChannels(FVTImage& img,const int ic0,const int ic1);

void _IFF_API fiCopyChannels(const FVTImage& src, const int icBeg,const int icEnd,
							FVTImage &dest,const int ocBeg);

void _IFF_API fiGetChannel(const FVTImage& src,FVTImage& dest,const int ic);

void _IFF_API fiColorToGray(const FVTImage& src,FVTImage& dest,
						   const double w0=0.114,const double w1=0.587,const double w2=0.299);

//convert channel numbers, gray to color, color to gray, 4-cn to 3-cn, 3-cn to 4-cn, etc.
//3,4->2 and 2->3,4 is not supported.

void _IFF_API fiConvertRGBChannels(const FVTImage &src,FVTImage &dest, const int ocn, double alpha=0);


void _IFF_API fiResize(const FVTImage &src, FVTImage &dest, int dwidth, int dheight, int resampleMethod=RESAMPLE_LINEAR);

void _IFF_API fiScale(const FVTImage &src, FVTImage &dest, double xscale, double yscale, int resampleMethod=RESAMPLE_LINEAR);




//==========================================================================================
//io functions

void _IFF_API fiLoadImage(const char_t* file,FVTImage& dest);

void _IFF_API fiSaveImage(const char_t* file,const FVTImage& image);

//==========================================================================================
//drawing functions

class _IFF_API FIBitmap
{
	class _CImp;

	_CImp *m_pimp;
public:
	FIBitmap();

	void SetImage(const FVTImage &img, bool flip);

	void Present(void *hdc, int dx=0, int dy=0, int ix=0, int iy=0, int cx=-1, int cy=-1);

	int  GetWidth() const;

	int  GetHeight() const;

	void*  GetBitmapHDC() const;

	void* GetBitmapBits() const;

	~FIBitmap();
};

_IFF_END


#endif

