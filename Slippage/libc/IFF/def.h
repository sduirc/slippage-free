

#ifndef _FF_IFF_DEF_H
#define _FF_IFF_DEF_H

#include <exception>

#include"ffdef.h"
#include"bfc\err.h"

#define _IFF_BEG _FF_BEG

#define _IFF_END _FF_END

#ifndef _IFF_API

#ifndef _IFF_STATIC

#ifdef IFF_EXPORTS
#define _IFF_API __declspec(dllexport)
#else
#define _IFF_API __declspec(dllimport)
#endif

#else

#define _IFF_API

#endif

#endif

_IFF_BEG

#define FI_MAKE_DEPTH(size,id) (((size)<<4)|(id))

//depth types, which encode the size of the type and a depth ID.
const int FI_8S=FI_MAKE_DEPTH(1,0);
const int FI_8U=FI_MAKE_DEPTH(1,1);
const int FI_16S=FI_MAKE_DEPTH(2,2);
const int FI_16U=FI_MAKE_DEPTH(2,3);
const int FI_32S=FI_MAKE_DEPTH(4,4);
const int FI_32F=FI_MAKE_DEPTH(4,5);

//invalid depth.
const int FI_INV_DEPTH=~0;
//number of depth types.
const int FI_DEPTH_NUM=6;

//determine whether the depth type is signed.
#define FI_DEPTH_SIGN_BIT(depth) (((depth)^1)&1)   
//retrive the size of the depth type.
#define FI_DEPTH_SIZE(depth)   (((depth)>>4)&0x0F) 
//retrive depth ID.
#define FI_DEPTH_ID(depth)		((depth)&0x0F)     

//make a pixel type from depth type and channel number.
#define FI_MAKE_TYPE(depth,cn)  (((cn)<<8)|depth)

const int FI_8SC1=FI_MAKE_TYPE(FI_8S,1);
const int FI_8SC2=FI_MAKE_TYPE(FI_8S,2);
const int FI_8SC3=FI_MAKE_TYPE(FI_8S,3);
const int FI_8SC4=FI_MAKE_TYPE(FI_8S,4);

const int FI_8UC1=FI_MAKE_TYPE(FI_8U,1);
const int FI_8UC2=FI_MAKE_TYPE(FI_8U,2);
const int FI_8UC3=FI_MAKE_TYPE(FI_8U,3);
const int FI_8UC4=FI_MAKE_TYPE(FI_8U,4);

const int FI_16SC1=FI_MAKE_TYPE(FI_16S,1);
const int FI_16SC2=FI_MAKE_TYPE(FI_16S,2);
const int FI_16SC3=FI_MAKE_TYPE(FI_16S,3);
const int FI_16SC4=FI_MAKE_TYPE(FI_16S,4);

const int FI_16UC1=FI_MAKE_TYPE(FI_16U,1);
const int FI_16UC2=FI_MAKE_TYPE(FI_16U,2);
const int FI_16UC3=FI_MAKE_TYPE(FI_16U,3);
const int FI_16UC4=FI_MAKE_TYPE(FI_16U,4);

const int FI_32SC1=FI_MAKE_TYPE(FI_32S,1);
const int FI_32SC2=FI_MAKE_TYPE(FI_32S,2);
const int FI_32SC3=FI_MAKE_TYPE(FI_32S,3);
const int FI_32SC4=FI_MAKE_TYPE(FI_32S,4);

const int FI_32FC1=FI_MAKE_TYPE(FI_32F,1);
const int FI_32FC2=FI_MAKE_TYPE(FI_32F,2);
const int FI_32FC3=FI_MAKE_TYPE(FI_32F,3);
const int FI_32FC4=FI_MAKE_TYPE(FI_32F,4);

//invalid pixel type.
const int FI_INV_TYPE=~0;
//number of predefined pixel types.
const int FI_TYPE_NUM=24;
//get channel number.
#define FI_CN(type)				 (((type)&0xFF00)>>8)
//get depth type from pixel type.
#define FI_DEPTH(type)			 ((type)&0xFF)
//type-id, from 0 to 23: {8SC1,8SC2,8SC3,8SC4,.....,32FC1,32FC2,32FC3,32FC4}.
#define FI_TYPE_ID(type)		 ((FI_DEPTH_ID(type)<<2)|(FI_CN(type)-1))
//channel size in bytes, equal to depth size.
#define FI_CHANNEL_SIZE(type)	 FI_DEPTH_SIZE(type)
//pixel size in bytes.
#define FI_TYPE_SIZE(type)		 (FI_CN(type)*FI_CHANNEL_SIZE(type))

#define FI_PIXEL_SIZE FI_TYPE_SIZE

/////////////////////////////////////////////////////
//constants and macros to define pixel formats.

#define FI_MAKE_FMT(fmt_id,type) (((fmt_id)<<16)|type)


const int FI_FMT_GRAY=FI_MAKE_FMT(1,FI_8UC1);
const int FI_FMT_ALPHA=FI_MAKE_FMT(2,FI_8UC1);
const int FI_FMT_GRAY_ALPHA=FI_MAKE_FMT(3,FI_8UC2);

const int FI_FMT_R8G8B8=FI_MAKE_FMT(4,FI_8UC3);
const int FI_FMT_B8G8R8=FI_MAKE_FMT(5,FI_8UC3);
const int FI_FMT_R8G8B8A8=FI_MAKE_FMT(6,FI_8UC4);
const int FI_FMT_B8G8R8A8=FI_MAKE_FMT(7,FI_8UC4);

//...other formats to be added.

const int FI_FMT_NUM=7;

const int FI_INV_FMT=~0;

//get pixel type from format.
#define FI_TYPE(fmt) ((fmt)&0xFFFF)

//get format ID.
#define FI_FMT_ID(fmt)   ((((fmt)&0x0F0000)>>16)-1)


//////////////////////////////////////////////////////////////////
//facility routines.

//to help implement function table which indexed by depth-ID.
#define FI_MAKE_DEPTH_FUNC_LIST(prefix,postfix) \
	prefix##char##postfix,prefix##uchar##postfix,\
	prefix##short##postfix,prefix##ushort##postfix,\
	prefix##int##postfix,prefix##float##postfix,\

#define FI_DEPTH_FUNC_LIST(func_template) FI_MAKE_DEPTH_FUNC_LIST(&func_template<,>)

//get accummulate-type according to depth type.
template<typename _Ty>
struct FI_ACCUM_TYPE
{
	typedef int type;
};

template<>
struct FI_ACCUM_TYPE<float>
{
	typedef float type;
};

//=====================================================
//type check and error handling

//whether a number is valid depth ID.
inline bool fiIsValidDepthID(const int id)
{
	return uint(id)<uint(FI_DEPTH_NUM);
}
//depth from depth-ID.
int _IFF_API fiDepthFromID(const int id);
//whether a depth is valid.
bool _IFF_API fiIsValidDepth(const int depth);

inline bool fiIsValidTypeID(const int id)
{
	return uint(id)<uint(FI_TYPE_NUM);
}
int _IFF_API fiTypeFromID(const int id);

bool _IFF_API fiIsValidType(const int type);

inline bool fiIsValidFormatID(const int id)
{
	return uint(id)<uint(FI_FMT_NUM);
}

int  _IFF_API fiFormatFromID(const int id);

bool _IFF_API fiIsValidFormat(const int fmt);


bool _IFF_API fiIsValidAlign(const int align);



#define FI_ASSERT_TYPE(type) assert(fiIsValidType(type)); 

#define FI_ASSERT_DEPTH(depth) assert(fiIsValidDepth(depth)); 

#define FI_ASSERT_FORMAT(format)  assert(fiIsValidFormat(format) ); 

#define FI_ASSERT_ALIGN(align) assert(fiIsValidAlign(align)); 

#define FI_ASSERT_STEP(width,type,step) assert((width)*FI_PIXEL_SIZE(type)<=(step)); 


_IFF_END

#endif

