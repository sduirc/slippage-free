


#include"libc-02.h"
#include"libd-01.h"


#ifdef  USE_VFX_LIB

#ifndef USE_WXWINDOWS_LIB
#define USE_WXWINDOWS_LIB
#endif

#endif

#include"libx-01.h"


#ifdef USE_VFX_LIB

#pragma comment(lib,_LN_DLL_LIB_AS(vfx))

#endif

#ifdef USE_VFXS_LIB

#pragma comment(lib,_LN_DLL_LIB_AS(vfxserv))

#endif

#ifdef USE_ARD_LIB
#pragma comment(lib,"ardcore.lib")
#endif

