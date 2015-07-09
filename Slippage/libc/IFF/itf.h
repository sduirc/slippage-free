
#ifndef _INC_IFF_ITF_H
#define _INC_IFF_ITF_H

#include"ffdef.h"
#include"iff/image.h"

_FF_BEG

enum
{
	FI_GAUSSIAN,
	FI_MEDIAN,
	FI_BLUR
};

void _BFC_API fiSmooth(const FVTImage &src,FVTImage &dest,int type=FI_GAUSSIAN,int wx=3,int wy=3);

void _BFC_API fiSmooth(FVTImage &img,int type=FI_GAUSSIAN,int wx=3,int wy=3);


void _BFC_API fiErode(const FVTImage &src,FVTImage &dest,int itr=1);

void _BFC_API fiErode(FVTImage &img,int itr=1);

void _BFC_API fiDilate(const FVTImage &src,FVTImage &dest,int itr=1);

void _BFC_API fiDilate(FVTImage &img,int itr=1);


enum
{
	FI_BGR2Luv=0,
	FI_Luv2BGR,

	FI_BGR2Lab,
	FI_Lab2BGR,

	FI_BGR2YCC,
	FI_YCC2BGR,

	FI_BGR2HSV,
	FI_HSV2BGR,

	FI_BGR2HLS,
	FI_HLS2BGR,

	FI_CVT_COLOR_END
};

void _BFC_API fiCvtColor(const FVTImage &src,FVTImage &dest,int cvt,int destDepth=FI_8U);

void _BFC_API fiCvtColor(FVTImage &img,int cvt,int destDepth=FI_8U);


//3 by 3 bilateral filter
void _BFC_API fiBilateral(const FVTImage &src,FVTImage &dest, int colorSigma,int spaceSigma);

void _BFC_API fiBilateral(FVTImage &img, int colorSigma,int spaceSigma);

void _BFC_API fiWarpAffine(const FVTImage &src,FVTImage &dest,const Matrix23f &mat,const Color4ub *pFillColor=NULL);

void _BFC_API fiWarpPerspective(const FVTImage &src,FVTImage &dest,const Matrix33f &mat,const Color4ub *pFillColor=NULL);

void _BFC_API fiGetAffineTransform(const Point2f pSrcPts[3],const Point2f pDestPts[3],Matrix23f &mat);

void _BFC_API fiGetPerspectiveTransform(const Point2f pSrcPts[4],const Point2f pDestPts[4],Matrix33f &mat);

void _BFC_API fiFindHomography(const Point2f *pSrcPts,const Point2f *pDestPts,int count,Matrix33f &hmMat);

//-----------------------------------------------------------------------------------
//drawing functions.

const int FI_LT_4C=4;
const int FI_LT_8C=8;
const int FI_LT_AA=10;

void _BFC_API fiCircle(FVTImage &img,const Point2i & center,int radius,const Color4ub &clrRGB,
					   int thick=1,int lineType=FI_LT_8C);

void _BFC_API fiLine(FVTImage &img,const Point2i & pt1,const Point2i &pt2,
					 const Color4ub &clrRGB,
					   int thick=1,int lineType=FI_LT_8C);

void _BFC_API fiRectangle(FVTImage &img,const Point2i &ltPt,const Point2i &brPt,const Color4ub &clrRGB,int thick=1,int lineType=FI_LT_8C);

void _BFC_API fiRectangle(FVTImage &img,const Rect &rect,const Color4ub &clrRGB,int thick=1,int lineType=FI_LT_8C);

void _BFC_API fiCross(FVTImage &img,const Point2i &pt,int radius,const Color4ub &clrRGB,int thick=1);

const int FI_FEA_CIRCLE=1;
const int FI_FEA_CROSS=2;

void _BFC_API fiDrawPoints(FVTImage &img,const Point2i *pPts,int nPts,int radius,const Color4ub &clrRGB,int thick=1,int feaType=FI_FEA_CIRCLE);

void _BFC_API fiDrawPoints(FVTImage &img,const Point2f *pPts,int nPts,int radius,const Color4ub &clrRGB,int thick=1,int feaType=FI_FEA_CIRCLE);



_FF_END



#endif

