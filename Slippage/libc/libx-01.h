
//#ifndef _INC_LIBX_01_H
//#define _INC_LIBX_01_H

#include"libname-01.h"

#define CXCORE_LIB 0x01
#define CV_LIB	   0x02
#define HIGHGUI_LIB 0x04
#define CVAUX_LIB	0x08
#define ML_LIB    0x10

#ifndef OPENCV_LIB_MASK 
#define OPENCV_LIB_MASK (CXCORE_LIB|CV_LIB|HIGHGUI_LIB|CVAUX_LIB|ML_LIB)
#endif


#if defined(USE_OPENCV_STATIC_LIB)&&(OPENCV_LIB_MASK&HIGHGUI_LIB)
#define USE_CODEC_LIB
#endif



//////////////////////////////////////////////////////
//opencv

#ifdef USE_OPENCV_LIB

#if OPENCV_LIB_MASK&CXCORE_LIB
#pragma comment(lib,"cxcore.lib")
#endif

#if OPENCV_LIB_MASK&CV_LIB
#pragma comment(lib,"cv.lib")
#endif

#if OPENCV_LIB_MASK&HIGHGUI_LIB
#pragma comment(lib,"highgui.lib")
#endif

#if OPENCV_LIB_MASK&CVAUX_LIB
#pragma comment(lib,"cvaux.lib")
#endif

#if OPENCV_LIB_MASK&ML_LIB
#pragma comment(lib,"ml.lib")
#endif

#endif


#ifdef USE_OPENCV_STATIC_LIB

#pragma comment(lib,_LN_STATIC_LIB_A(OpenCV))
#pragma comment(lib,"vfw32.lib")
#pragma comment(lib,"comctl32.lib")

#endif

#define _LN_OPENCV2(name) "opencv_"##_LN_STR(name)##"240"##_LN_DEBUG()##".lib"

#ifdef USE_OPENCV2_LIB

#pragma comment(lib,_LN_OPENCV2(core))
#pragma comment(lib,_LN_OPENCV2(features2d))
#pragma comment(lib,_LN_OPENCV2(imgproc))
#pragma comment(lib,_LN_OPENCV2(highgui))
#pragma comment(lib,_LN_OPENCV2(gpu))
#pragma comment(lib,_LN_OPENCV2(calib3d))
#pragma comment(lib,_LN_OPENCV2(imgproc))
#pragma comment(lib,_LN_OPENCV2(ml))
#pragma comment(lib,_LN_OPENCV2(video))
#pragma comment(lib,_LN_OPENCV2(objdetect))
#pragma comment(lib,_LN_OPENCV2(flann))

#endif


/////////////////////////////////////////////////////
//direct-show
#ifdef USE_DIRECTSHOW_LIB

#pragma comment(lib,"dxerr9.lib")
#pragma comment(lib,"strmiids.lib")
#pragma comment(lib,"winmm.lib")
#pragma comment(lib,"atls.lib")

#pragma comment(lib,_LN_STATIC_LIB_A(Strmbas))

#endif //_DEBUG



//////////////////////////////////////////////////
//direct-3d

#ifdef USE_DIRECT3D_LIB

//dxerr9.lib dxguid.lib d3dx9d.lib d3d9.lib winmm.lib comctl32.lib

#pragma comment(lib,"dxerr9.lib")
#pragma comment(lib,"dxguid.lib")
#pragma comment(lib,"winmm.lib")
#pragma comment(lib,"comctl32.lib")
#pragma comment(lib,"d3d9.lib")
#pragma comment(lib,"atls.lib")

#ifdef _DEBUG
#pragma comment(lib,"d3dx9d.lib")
#else
#pragma comment(lib,"d3dx9.lib")
#endif

#endif



////////////////////////////////////////////////////
//opengl

#ifdef USE_OPENGL_LIB

#pragma  comment(lib,"opengl32.lib")
#pragma  comment(lib,"glu32.lib")

#endif


//wxwindows
#ifdef USE_WXWINDOWS_LIB


#ifdef _DEBUG
#pragma comment(lib,"wxmsw28d.lib")
#else
#pragma comment(lib,"wxmsw28.lib")
#endif

#endif //end of wxwindows


//////////////////////////////////////////////////////
//codec
#ifdef  USE_CODEC_LIB
#pragma comment(lib,_LN_STATIC_LIB_A(Codec))
#endif

