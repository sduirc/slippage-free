
#ifndef _FF_IFF_UTIL_H
#define _FF_IFF_UTIL_H

#include"iff\def.h"

_IFF_BEG


//cast @value to the type denoted by @depth (FI_8S,FI_8U,....) and then store
//the result to the memory addressed by @pDest.
//@count : number of elements in @pDest (each element is assigned the same value).
void _IFF_API  fiuFillBits(void *pDest,int depth,int count,double value);

//copy image to another buffer.
//@rowSize : bytes of data to copy in each line, which maybe less or equal to @istep.
void  _IFF_API  fiuCopy(const void* pIn,const int rowSize,const int height,const int istep,
			  void* pOut,const int ostep);

//copy certain part of each pixel of the input image into another image, for example, copy the
//second channel of a color image into a gray image.

//@ips : the size of each pixel in input image.
//@ops : the size of each pixel in output image.
//@cps : size of the data to copy for each pixel.

void _IFF_API fiuCopyPixel(const void *pIn,const int width,const int height,const int istep,const int ips,
						 void *pOut,const int ostep,const int ops,const int cps);

//rotate counter-clockwise for 90*n degrees
void _IFF_API fiuRotate90(const void *pIn, const int width, const int height, const int istep, void *pOut, const int ostep, const int ps, const int n90=1);

//set each character of image data to @val.
void  _IFF_API fiuSetMem(void* pOut,const int rowSize,const int height,const int step,const char val);

//set boudary of image data.
void _IFF_API  fiuSetBoundary(void *pOut,const int rowSize,const int height,const int step,
							  const int nTopLines,const int nBottomLines,
							  const int nLeftBytes,const int nRightBytes,
							  const char val
							  );

//fill each pixel of an image with the contents in the buffer addressed by @pval, whose size is @cps.
//Note that @cps may be less than the pixel size @ps.
void  _IFF_API fiuSetPixel(void* pOut,const int width,const int height,const int step,
				  const int ps,const void* pval,const int cps);

void  _IFF_API  fiuSetBoundaryPixel(void *pOut,const int width,const int height,const int step,
									const int ps,const void *pval,const int cps,
									const int nTopLines,const int nBottomLines,
									const int nLeftPixels,const int nRightPixels
									);

//flip with respect to horizontal axis.
void  _IFF_API  fiuFlipX(const void* pIn,const int lineSize,const int height,const int istep,
			  void* pOut,const int ostep);

void _IFF_API   fiuFlipX(void* pInOut,const int lineSize,const int height,const int step);

//flip with respect to vertical axis.
void _IFF_API  fiuFlipY(const void* pIn,const int width,const int height,const int istep,const int pixelSize,
		  void* pOut,int ostep);

void _IFF_API  fiuFlipY(void* pInOut,const int width,const int height,const int istep,const int pixelSize);

//repeat image @pIn in output buffer @pOut.
void _IFF_API  fiuTile(const void* pIn,const int iwidth,const int iheight,const int istep,
					  void* pOut,const int owidth,const int oheight,const int ostep,
					  const int pixelSize);

//transform each pixel as : y=@scale*x+@shift, where x is the intensity of input image and y is the
//result.
void _IFF_API  fiuTransformPixel(const void* pIn,const int width,const int height,const int istep,const int itype,
	   void* pOut,const int ostep,const int odepth,const double scale=1,const double shift=0);

//swap two channels of th given image.

//@ic0,@ic1: the zero-based ID of the two channels.
void _IFF_API  fiuSwapChannels(void *pImg,const int width,const int height,const int step,
							  const int type,const int ic0,const int ic1);

void _IFF_API  fiuColorToGray(const void* pIn,const int width,const int height,const int istep,const int type,
							void* pOut,const int ostep,
							const double w0,const double w1,const double w2);

//convert gray image to color image.
//@alpha : the value to fill alpha channel.

void _IFF_API  fiuGrayToColor(const void* pGray,const int width,const int height,const int istep,const int type,
					void* pColor,const int ostep,const int ocn,double alpha);

//copy some continuous channels from one image to another.
//@icBeg,@icEnd : the first and last (not included in) channel to copy.
//@ocn : the channel number of output image.
void _IFF_API fiuCopyChannels(const void* pSrc,const int width,const int height,const int istep,const int type,
					 const int icBeg,const int icEnd,
					 void* pDest,const int ostep,const int ocn,const int ocBeg);

//convert the channel number of image in RGB space.
void _IFF_API fiuConvertRGBChannels(const void *pIn,const int width,const int height,const int istep,const int type,
							 void *pDest, const int ostep, const int ocn,double alpha=0);

enum
{
	RESAMPLE_NN=1,
	RESAMPLE_LINEAR,
	RESAMPLE_CUBIC
};

void _IFF_API fiuResize(const void *src, int width, int height, int istep, int type, void *dest, int dwidth, int dheight, int dstep, int resampleMethod=RESAMPLE_LINEAR);

//mask connected component.
//A component is a maximum 8-connected region whose pixels have the same intensity.
//@odetph: type of the output masks.
//@bkVal: the intensity of background pixels. All the background pixels are masked with zero and thus not be further divided.
//return the number of components, including background.
int _IFF_API fiuConnectedComponent(const uchar *pSrc,const int width,const int height,const int istep,
									void *pCC,const int ostep,const int odepth,
									const int bkVal,bool bC8=true
									);


_IFF_END

#endif

