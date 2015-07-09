
#ifndef _FF_FFDEF_H
#define _FF_FFDEF_H



#ifndef _UINT_DEFINED
#define _UINT_DEFINED

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

typedef uint	uint32;
typedef int		int32;

#endif 

#ifdef _UNICODE

#define string_t std::wstring
#define char_t   wchar_t

#define ifstream_t std::wifstream
#define istream_t  std::wistream

#define _TX(x) L ## x

#else

#define string_t std::string
#define char_t   char

#define ifstream_t std::ifstream
#define istream_t  std::istream

#define _TX(x) x

#endif

#define _FF_BEG  namespace ff{

#define _FF_END }

#define _FF_NS(x) ff::x

#pragma warning(disable:4996)

#define _BFC_API

//define _FFS_API
#ifndef _FFS_API

#ifndef _FFS_STATIC

#ifdef FFS_EXPORTS
#define _FFS_API __declspec(dllexport)
#else
#define _FFS_API __declspec(dllimport)
#endif

#else

#define _FFS_API

#endif

#endif

//...

#define FF_COPYF_DECL(class_name) class_name(const class_name&); class_name& operator=(const class_name &);

#endif

