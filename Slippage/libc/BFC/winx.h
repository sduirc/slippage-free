
#ifndef _FF_BFC_WINX_H
#define _FF_BFC_WINX_H

#include<string>
#include<vector>

#include<windows.h>

#include"bfc\type.h"

_FF_BEG


#define FF_VK_CHAR(cc) (0x41+(cc)-'A')

#define _VKC(cc) FF_VK_CHAR(cc)

#define FF_VK_NUM(nc)  (0x30+(nc)-'0')

#define _VKN(nc) FF_VK_NUM(nc)

//get error description of the error code return by ::GetLastError()
string_t _BFC_API wxGetErrorString(DWORD ec);

string_t _BFC_API wxMakeErrorString(DWORD ec,const char_t * pMsg);

/////////////////////////////////////////////////////////////////////////

//Initialize the console window and enable standard c/c++ I/O streams.
//Typically used in GUI application without console

void _BFC_API wxInitConsole();

enum tagConsole{STD_INPUT=0,STD_OUTPUT=1,STD_ERROR=2};

Point2i _BFC_API wxGetStdConsoleCursorPos(tagConsole tag=STD_OUTPUT);

void  _BFC_API wxSetStdConsoleCursorPos(const Point2i &pos,tagConsole tag=STD_OUTPUT);


//get current directory, the returned string is always end with a backslash '\'.
string_t _BFC_API wxGetCurrentDir();

//set current directory to @dir.
void		_BFC_API wxSetCurrentDir(const string_t& dir);

BOOL wxDeleteFile(const string_t &file);

BOOL wxListFiles(const string_t &dir, std::vector<WIN32_FIND_DATA> &vfiles);

BOOL wxRemoveDirectory(const string_t &dir);

//Help to initialize a @BITMAPINFO struct.
//@width,@height: size of the bitmap.
//@nBits : color bits per-pixel.
//@compression : compression method.
void _BFC_API wxInitBitmapInfo(BITMAPINFO& bmh,int width,int height,uint nBits=32,int compression=BI_RGB);

//Create a DDB device compatible with the specified device.
//@width,@height: the size of the DDB to create.
//@hdc : the handle of the device to specify the DDB format, that is, the created DDB must compatible with
//		this device. The default value NULL specify current desktop to be such a device.
HDC _BFC_API wxCreateDeviceInDDB(unsigned width,unsigned height,HDC hdc=NULL);

//Create a DIB section with 8 color bits.
//@width,@height: the size of the DIB to create.
//@ppv : a pointer to pointer which is used to receive the beginning address of the data block allocated for the DIB.
//@pallete: a pointer to a 256-color table which contain pallete entries for the DIB to use,
//			the default value NULL cause gray pallete be used.
//Return value: the handle to the created DIB.
HBITMAP _BFC_API wxCreateDIBSection8b(unsigned width,unsigned height,void** ppv=0,const RGBQUAD* palette=0);

//Create a DIB section.
//@width,@height,@ppv : see @wxCreateDIBSection8b.
//@nBits : specify the color bits of the DIB, if this parameter is 8, the @wxCreateDIBSection8b will be called
//			to create a DIB with gray pallete.
HBITMAP _BFC_API wxCreateDIBSection(unsigned width,unsigned height,void** ppv=0,unsigned nBits=32);

//create a compatible memroy device in bitmap.
HDC _BFC_API wxCreateDeviceInBitmap(HBITMAP hbmp);

//Create a memory device in DIB.
//this function will create the required DIB and a compatible memory device, and then select the
//bitmap into the device.
HDC _BFC_API wxCreateDeviceInDIB(unsigned width,unsigned height,void** ppv=0,unsigned nBits=32);

//delete a memory compatible device, which is typically created by @wxCreateDeviceInDDB or @wxCreateDeviceInDIB.
//note this function will delete the bitmap selected into the device.
void _BFC_API wxDeleteBitmapDevice(HDC hdc);

//get the largest display mode supported by current display, which is useful to create a large
//enough bitmap to save the screen.
void _BFC_API wxGetLargestDisplayMode(unsigned *pWidth,unsigned *pHeight);

/////////////////////////////////////////////////////////////////////////////////
//The following two functions can be used to create and destroy a hide window to initialize 
//OpenGL, Direct3D or some other libraries which can't be initialized without window.

HWND _BFC_API wxCreateDevWindow(int width,int height,bool bVisible=false);

void _BFC_API wxDestroyDevWindow(HWND hWnd);

//////////////////////////////////////////////////////////////////////////////
//utility functions to resize window.

//get the window size.
void _BFC_API wxGetWindowSize(HWND hwnd,int *pWidth,int *pHeight);

//get only the window client size.
void _BFC_API wxGetClientSize(HWND hwnd,int *pWidth,int *pHeight);

//get the size of the window's non-client region.
void _BFC_API wxGetNCSize(HWND hwnd,int *pWidth,int *pHeight);

//resize a window without change its left-top position.
//@bRepaint: whether to repaint the window after resize.
void _BFC_API wxResizeWindow(HWND hwnd,int width,int height,bool bRepaint);

//similar to @wxResizeWindow but the parameter @width,@height is used to specify the size
//of the client area.
void _BFC_API wxResizeWindowClient(HWND hwnd,int width,int height,bool bRepaint);

/////////////////////////////////////////////////////////////////////////////

void _BFC_API wxSetProcessPriority(int priority=-1);


/////////////////////////////////////////////////////////////////////////////
//drawing functions

const int PS_RECT=1;
const int PS_FILL_RECT=2;
const int PS_CIRCLE=3;
const int PS_FILL_CIRCLE=4;

void _BFC_API wxDrawPoint(HDC hdc,int x,int y,int style=PS_FILL_CIRCLE,int radius=3);

void _BFC_API wxDrawPointEx(HDC hdc,int x,int y,COLORREF clr,int style=PS_FILL_CIRCLE,int radius=3);


const int LS_SOLID=PS_SOLID;
const int LS_DASH =PS_DASH;
const int LS_DOT  =PS_DOT;
const int LS_DASHDOT=PS_DASHDOT;
const int LS_DASHDOTDOT=PS_DASHDOTDOT;

void _BFC_API wxDrawLine(HDC hdc,const POINT& start,const POINT& end);

void _BFC_API wxDrawLineEx(HDC hdc,const POINT& start,const POINT& end,COLORREF clr=RGB(0,0,0),int style=LS_SOLID,int width=1);

const int FLS_CLOSED=0x01;

void _BFC_API wxDrawFreeLine(HDC hdc,const POINT* points,size_t count,int flsStyle=0);

void _BFC_API wxDrawFreeLineEx(HDC hdc,const POINT* points,size_t count,int flsStyle=0,
							   COLORREF clr=RGB(0,0,0),int lineStyle=LS_SOLID,int width=1);



_FF_END


#endif
