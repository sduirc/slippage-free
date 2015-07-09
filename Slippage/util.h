
#pragma once

#include"ffdef.h"


#define _CV_WH(mat)   (mat).cols,(mat).rows

#define _CV_DWHS(mat)  (mat).data,_CV_WH(mat),(mat).step

#define _CV_DWHSC(mat)  _CV_DWHS(mat),(mat).channels()

#define _CV_DSC(mat)  (mat).data,(mat).step,(mat).channels()
