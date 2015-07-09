
#ifndef _FF_INC_FIOPENCV_H
#define _FF_INC_FIOPENCV_H

#ifdef _MSC_VER
#pragma warning(disable: 4819)
#endif

#include<cv.h>

#ifdef _MSC_VER
#pragma warning(default:4819)
#endif

#include"iff/image.h"
#include"iff/util.h"
#include"ffdef.h"

_FF_BEG

int _BFC_API fiDepthToCV(const int fi_depth);

int _BFC_API fiTypeToCV(const int fi_type);

int _BFC_API fiDepthFromCV(const int cv_depth);

int _BFC_API fiTypeFromCV(const int cv_type);

int _BFC_API fiDepthToIPL(const int fi_depth);

class FICvMat
	:public CvMat
{
public:
	FICvMat()
	{
	}
	FICvMat(const FVTImage& fimg);

};

inline CvSize fiCvSize(const FVTImage& fimg)
{
	return cvSize(fimg.Width(),fimg.Height());
}

class FIIplImage
	:public IplImage
{
public:
	FIIplImage()
	{
	}
	FIIplImage(const FVTImage& fimg);
};


//////////////////////////////////////////////////////////////////


_FF_END


#endif





