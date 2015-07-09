
#ifndef _FF_IFF_IOIMPL_H
#define _FF_IFF_IOIMPL_H

#include"iff\image.h"
#include<tchar.h>

using namespace ff;

const int FI_IO_IGNORE_ALPHA=0x01;

const int QIM_SWAP_RB=0x01;
const int QIM_DITHER=0x02;

int _IFF_API  QuantizeImageMedian(const uchar *pData,const int width,const int height,const int step,const int cn,
				   uchar* pIndex,uchar (*pMap)[3],const int mapSize,int mode=QIM_SWAP_RB|QIM_DITHER);


void _IFF_API  fiSaveGif(const char_t* file,const FVTImage& img,const uchar pTransparent[3]=NULL);

void _IFF_API fiLoadImageImpl(const char_t* file, FVTImage& image);

void _IFF_API fiSaveImageImpl(const char_t* file,const FVTImage& image);

#endif

