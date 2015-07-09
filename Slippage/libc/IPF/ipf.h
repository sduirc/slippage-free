
#ifndef _FF_IPF_IPF_H
#define _FF_IPF_IPF_H


#include"ipf\def.h"
#include"ipf\ipt.h"

_IPF_BEG


_IPF_API void  ipf_memcpy_2d(const void* src, int row_size, int height, int istep, void* dest, int ostep);

_IPF_API void  ipf_memcpy_3d(const void *src, int row_size, int height, int length, int iystep, int izstep, void *dest, int oystep, int ozstep);

_IPF_API void  ipf_memset_2d(void* buf,const int row_size,const int height,const int step, char val);

_IPF_API void  ipf_memset_3d(const void *buf, int row_size, int height, int length, int ystep, int zstep, char val);

_IPF_API void  ipf_flip_vert(const void* src,const int row_size,const int height,const int istep, void* dest,const int ostep);

_IPF_API void  ipf_flip_vert(void* img,const int row_size,const int height,const int step);

_IPF_API void  ipf_flip_horz(const void* src,const int width,const int height,const int istep,const int pixel_size, void* dest,int ostep);

_IPF_API void  ipf_flip_horz(void* img,const int width,const int height,const int istep,const int pixel_size);


template<typename _PValT>
inline typename pval_diff_type<_PValT>::type px_diff_c3(const _PValT *px, const _PValT *py)
{
	typedef typename pval_diff_type<_PValT>::type _DiffT;

	_DiffT dx[]={_DiffT(px[0])-_DiffT(py[0]),_DiffT(px[1])-_DiffT(py[1]),_DiffT(px[2])-_DiffT(py[2])};

	return dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2];
}

template<typename _PValT>
inline typename pval_diff_type<_PValT>::type px_diff_c4(const _PValT *px, const _PValT *py)
{
	typedef typename pval_diff_type<_PValT>::type _DiffT;

	_DiffT dx[]={_DiffT(px[0])-_DiffT(py[0]),_DiffT(px[1])-_DiffT(py[1]),_DiffT(px[2])-_DiffT(py[2]), _DiffT(px[3])-_DiffT(py[3])};

	return dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]+dx[3]*dx[3];
}


//@ccn : number of channels to be copied

template<typename _PValT, typename _CNT>
inline void ipf_copy_channels(const _PValT *src, int width, int height, int istride, int icn, _PValT *dest, int ostride, int ocn, _CNT acn)
{
	for_each_pixel_1_1(src,width,height,istride,icn,dest,ostride,ocn, iop_copy<cnt<_CNT>::CNT>(acn));
}

template<typename _PValT, typename _CNT>
inline void ipf_set_channels(_PValT *img, int width, int height, int stride, int cn, const _PValT *pv, _CNT acn)
{
	for_each_pixel_0_1(img,width,height,stride,cn,bind_i0(pv,iop_copy<cnt<_CNT>::CNT>(acn)));
}

template<typename _PValT, typename _CNT>
inline void ipf_max(const _PValT *img, int width, int height, int stride, const int cn, _PValT _max[], _CNT acn)
{
	pxcpy(img,_max,acn);

	for_each_pixel_1_0(img,width,height,stride,cn,iop_max<cnt<_CNT>::CNT,_PValT>(_max,acn) );
}

template<typename _PValT, typename _CNT>
inline void ipf_min(const _PValT *img, int width, int height, int stride, const int cn, _PValT _min[], _CNT acn)
{
	pxcpy(img,_min,acn);

	for_each_pixel_1_0(img,width,height,stride,cn,iop_min<cnt<_CNT>::CNT,_PValT>(_min,acn) );
}

template<typename _PValT, typename _CNT>
inline void ipf_min_max(const _PValT *img, int width, int height, int stride, const int cn, _PValT _min[], _PValT _max[], _CNT acn)
{
	pxcpy(img,_max,acn);
	pxcpy(img,_min,acn);

	for_each_pixel_1_0(img,width,height,stride,cn,iop_min_max<cnt<_CNT>::CNT,_PValT>(_min,_max,acn) );
}

template<typename _PValT, typename _HValT>
inline void ipf_get_histogram(const _PValT *img, int width, int height, int istride, int icn, _HValT hist[], int nbin, bool init=true)
{
	if(init)
		memset(hist,0,sizeof(_HValT)*nbin);

	for(int yi=0; yi<height; ++yi, img+=istride)
	{
		const _PValT *pix=img;

		for(int xi=0; xi<width; ++xi, pix+=icn)
			hist[*pix]+=1;
	}
}

// <T0: B, >=T1: F, otherwise: U
inline void ipf_threshold(const uchar *src, int width, int height, int istep, uchar *dest, int dstep, uchar T0, uchar T1, uchar v0, uchar vx, uchar v1)
{
	if(src && dest)
	{
		uchar mapx[256];

		if(T0!=0)
			memset(&mapx[0],v0,T0);

		if(T0!=T1)
			memset(&mapx[T0],vx,T1-T0);

		if(T1<=255)
			memset(&mapx[T1],v1,256-T1);

		for_each_pixel_1_1(src,width,height,istep,1,dest,dstep,1,iop_map<1,uchar>(mapx));
	}
}

template<typename _PValT, typename _FValT, typename _CNT>
inline void ipf_warp_by_flow_nn(const _PValT *src, int width, int height, int sstride, _CNT scn, _PValT *dest, int dstride, const _FValT *reverse_flow_xy, int fstride)
{
	for(int yi=0; yi<height; ++yi, dest+=dstride, reverse_flow_xy+=fstride)
	{
		_PValT *pdx=dest;
		for(int xi=0; xi<width; ++xi, pdx+=scn)
		{
			int x=int(xi+reverse_flow_xy[xi*2]+0.5), y=int(yi+reverse_flow_xy[xi*2+1]+0.5);
			if(uint(x)>=uint(width)||uint(y)>=uint(height))
			{
				x=xi; y=yi;
			}

			pxcpy(src+y*sstride+x*scn,pdx,scn);
		}
	}
}

template<typename _PValT, typename _DPtrT>
inline void ipf_accumulate_2d(const _PValT *src, int width, int height, int sstride, int cn, _DPtrT &sum, _PValT init_val=0)
{
	for(int i=0; i<cn; ++i)
		sum[i]=init_val;

	for(int yi=0; yi<height; ++yi, src+=sstride)
	{
		const _PValT *psx=src;
		for(int xi=0; xi<width; ++xi, psx+=cn)
		{
			for(int i=0; i<cn; ++i)
				sum[i]+=psx[i];
		}
	}
}

//===========================================================

_IPF_API void ipf_calc_resize_tab_nn(int ssize, int dsize, int tab[], int cn);


template<typename _PValT, typename _CNT>
inline void ipf_resize_nn(const _PValT *src, int width, int height, int istride, const int icn, _PValT *dest, int dwidth, int dheight, int ostride, const int ocn, _CNT cn)
{
	if(width==dwidth && height==dheight && icn==cn && ocn==cn)
		ipf_memcpy_2d(src,width*sizeof(_PValT)*cn,height,sizeof(_PValT)*istride,dest,sizeof(_PValT)*ostride);
	else
	{
		int *xt=new int[dwidth], *yt=new int[dheight];

		ipf_calc_resize_tab_nn(width,dwidth,xt,icn);
		ipf_calc_resize_tab_nn(height,dheight,yt,1);

		for(int yi=0; yi<dheight; ++yi, dest+=ostride)
		{
			const _PValT *psy=src+istride*yt[yi];
			_PValT *pdx=dest;

			for(int xi=0; xi<dwidth; ++xi,pdx+=ocn)
			{
				pxcpy(psy+xt[xi], pdx, cn);
			}
		}

		delete[]xt;
		delete[]yt;
	}
}

template<typename _PValT, typename _CNT>
inline void ipf_resize_nn(const _PValT *src, int width, int height, int istride, _PValT *dest, int dwidth, int dheight, int ostride, _CNT cn)
{
	ipf_resize_nn(src,width,height,istride,cn,dest,dwidth,dheight,ostride,cn,cn);
}


_IPF_API void ipf_calc_resize_tab_bl(int ssize, int dsize, int itab[], float ftab[], int cn);


template<typename _IPValT, typename _OPValT>
inline void ipf_resize_bl(const _IPValT *src, int width, int height, int istride, const int icn, _OPValT *dest, int dwidth, int dheight, int ostride, const int ocn, const int cn)
{
	float *xt=new float[dwidth], *yt=new float[dheight];
	int  *ixt=new int[dwidth], *iyt=new int[dheight];

	ipf_calc_resize_tab_bl(width,dwidth,ixt,xt,icn);
	ipf_calc_resize_tab_bl(height,dheight,iyt,yt,1);

	for(int yi=0; yi<dheight; ++yi, dest+=ostride)
	{
		const _IPValT *psy=src+istride*iyt[yi];
		const float ry=yt[yi];

		_OPValT *pdx=dest;

		for(int xi=0; xi<dwidth; ++xi,pdx+=ocn)
		{
			const _IPValT *pi=psy+ixt[xi];
			const float rx=xt[xi];

			for(int i=0; i<cn; ++i, ++pi)
			{
				const _IPValT *pix=pi;

				float tx0=float(*pix+(*(pix+icn)-*pix)*rx);
				
				pix+=istride;
				float tx1=float(*pix+(*(pix+icn)-*pix)*rx);

				pdx[i]=static_cast<_OPValT>(tx0+(tx1-tx0)*ry);
			}
		}
	}

	delete[]xt;
	delete[]yt;
	delete[]ixt;
	delete[]iyt;
}


template<typename _IPValT, typename _OPValT>
inline void ipf_resize_bl(const _IPValT *src, int width, int height, int istride, _OPValT *dest, int dwidth, int dheight, int ostride, const int cn)
{
	ipf_resize_bl(src,width,height,istride,cn,dest,dwidth,dheight,ostride,cn,cn);
}

_IPF_API void ipf_calc_nbr_offset(int hw, int hh, int step, int cn, int *offset, bool exclude_me=true);

template<typename _ValT>
inline void ipf_clip_window_1d(_ValT &window_begin, _ValT &window_end, _ValT image_begin, _ValT image_end)
{
	if(window_begin<image_begin)
		window_begin=image_begin;

	if(window_end>image_end)
		window_end=image_end;
}


template<typename _IValT, typename _DValT, typename _KValT, int cn>
inline void ipf_convolve(const _IValT *data, ff::ccn<cn> _cn, int px_stride, const _KValT kw[], int size, _DValT dest[], int shift)
{
	_KValT dv[cn]={0};

	for(int i=0; i<size; ++i, data+=px_stride)
	{
		for(int j=0; j<cn; ++j)
			dv[j]+=data[j]*kw[i];
	}
	for(int i=0; i<cn; ++i)
		dest[i]=_DValT(dv[i]>>shift);
}

template<typename _IValT, typename _DValT, typename _KValT, typename _CNT>
inline void ipf_smooth_1d(const _IValT *data, int count, _CNT cn, int px_stride, _DValT *dest, int dpx_stride, const _KValT kw[], int ksz, int shift)
{
	const int hksz=ksz/2;

	_DValT *dx=dest;

	for(int i=1; i<=hksz; ++i, dx+=dpx_stride)
	{
		ipf_convolve(data,cn,px_stride, kw+hksz-i, hksz+i+1, dx, shift);
	}
	
	const _IValT *px=data;
	for(int i=hksz*2; i<count; ++i, px+=px_stride, dx+=dpx_stride)
	{
		ipf_convolve(px,cn,px_stride,kw,ksz,dx,shift);
	}

	for(int i=1; i<=hksz; ++i, px+=px_stride, dx+=dpx_stride)
	{
		ipf_convolve(px,cn,px_stride,kw,ksz-i,dx,shift);
	}
}

template<typename _IValT, typename _DValT, typename _KValT, typename _CNT>
inline void ipf_smooth(const _IValT *img, int width, int height, int istride, _CNT cn, _DValT *dimg, int dstride, const _KValT kw[], int ksz, int shift)
{
	_KValT *buf=new _KValT[width*cn*height];

	for(int xi=0; xi<width; ++xi)
	{
		ipf_smooth_1d(img+xi*cn,height,cn,istride,buf+xi*cn,width*cn,kw,ksz,shift);
	}
	for(int yi=0; yi<height; ++yi)
	{
		ipf_smooth_1d(buf+yi*width*cn,width,cn,cn,dimg+dstride*yi,cn,kw,ksz,shift);
	}

	delete[]buf;
}

template<typename _KValT>
inline void ipf_make_gaussian_kernel(double sigma, _KValT kw[], int ksz)
{
	const int hksz=ksz/2;
	sigma=2*sigma*sigma;
	_KValT sum=0;
	for(int i=0; i<ksz; ++i)
	{
		const int d=i-hksz;
		kw[i]=exp(-d*d/sigma);
		sum+=kw[i];
	}
	for(int i=0; i<ksz; ++i)
		kw[i]/=sum;
}

template<typename _KValT>
inline void ipf_make_gaussian_kernel_fp(double sigma, _KValT kw[], int ksz, int shift)
{
	double *dkw=new double[ksz];
	ipf_make_gaussian_kernel(sigma,dkw,ksz);

	for(int i=0; i<ksz; ++i)
		kw[i]=_KValT( dkw[i]*(1<<shift) + 0.5);

	delete[]dkw;
}

template<typename _IValT, typename _DValT, typename _CNT>
inline void ipf_smooth_gaussian(const _IValT *img, int width, int height, int istride, _CNT cn, _DValT *dimg, int dstride, int ksz, double sigma)
{
	const int shift=12;
	int *kw=new int[ksz];
	ipf_make_gaussian_kernel_fp(sigma,kw,ksz,shift);
	ipf_smooth(img,width,height,istride,cn,dimg,dstride,kw,ksz,shift);
	delete[]kw;
}

//===========================================================

//return 0 if ROI is empty

//get the range of all masked pixels
_IPF_API int ipf_get_mask_range(const uchar *mask, int count, int &imin, int &imax);

//get the bouding rect. of all masked pixels
_IPF_API int ipf_get_mask_roi(const uchar *mask, int width, int height, int mstep, int LTRB[], uchar T=0);

//get the bounding rect. of pixels with different mask value
_IPF_API int ipf_get_mask_diff_roi(const uchar *maskx, int width, int height, int xstep, const uchar *masky, int ystep, int LTRB[]);



//===========================================================
//Van Herk fast max/min filter

//!@ksz muast be odd
//!@dest can share the same memory with @src
_IPF_API void ipf_min_filter(const uchar *src, int width, int height, int istep, int cn, uchar *dest, int dstep, int ksz, int nitr=1, int borderLTRB[] =NULL);

_IPF_API void ipf_max_filter(const uchar *src, int width, int height, int istep, int cn, uchar *dest, int dstep, int ksz, int nitr=1, int borderLTRB[] =NULL);


//===========================================================
//fast sum filter implemented with integral image

_IPF_API void ipf_sum_filter(const uchar *src, int width, int height, int istep, int cn, int *dest, int dstride, int kszx, int kszy, int borderLTRB[]);

_IPF_API void ipf_sum_filter(const int *src, int width, int height, int istep, int cn, int *dest, int dstride, int kszx, int kszy, int borderLTRB[]);

//count the number of pixels involved at each pixel
_IPF_API void ipf_count_filter_pixels(int *np, int width, int height, int nstride, int kszx, int kszy, int borderLTRB[]);

//mean filter
//@np can be NULL if the number of involved pixels is not necessary
_IPF_API void ipf_mean_filter(const uchar *src, int width, int height, int istep, int cn, int *dest, int dstride, int *np, int nstride, int kszx, int kszy, int borderLTRB[]);

_IPF_API void ipf_mean_filter(const int *src, int width, int height, int istep, int cn, int *dest, int dstride, int *np, int nstride, int kszx, int kszy, int borderLTRB[]);

_IPF_API void ipf_mean_filter(const uchar *src, int width, int height, int istep, int cn, uchar *dest, int dstride, int *np, int nstride, int kszx, int kszy, int borderLTRB[]);

//===========================================================

/*
@cc : output the zero-based index of the connected component ( 4-connected pixels with the same mask value)

return the number of connected components
*/

_IPF_API int ipf_connected_component(const uchar *mask,const int width,const int height,const int mstep, int *cc,const int cstride);


//===========================================================
//distance transform functions

struct dist_field;

_IPF_API dist_field* ipf_create_distance_field_2d(int width, int height, ushort dist_scale=100);

_IPF_API dist_field* ipf_create_distance_field_3d(int width, int height, int length, ushort dist_scale=100);

_IPF_API void ipf_release_distance_field(dist_field *df);


_IPF_API int ipf_distance_transform_2d(dist_field *df, const uchar *mask, ushort width, ushort height, int mstep, uchar mvs, uint *dist, int dstride, ushort nxy[][2], int nstride);

_IPF_API int ipf_distance_transform_3d(dist_field *df, const uchar *mask, ushort width, ushort height, ushort length, uchar mvs, uint *dist, ushort nxyz[][3]);


//===========================================================
//color space transformation

_IPF_API void ipf_bgr2lab(const uchar *bgr, uchar *lab);

_IPF_API void ipf_lab2bgr(const uchar *lab, uchar *bgr);

_IPF_API void ipf_bgr2yuv(const uchar *bgr, uchar *yuv);

_IPF_API void ipf_yuv2bgr(const uchar *yuv, uchar *bgr);

//_IPF_API void ipf_bgr2lab(const uchar *bgr, int width, int height, int istep, int icn, uchar *lab, int dstep, int dcn);

//_IPF_API void ipf_lab2bgr(const uchar *lab, int width, int height, int istep, int icn, uchar *bgr, int dstep, int dcn);



_IPF_END


#endif

