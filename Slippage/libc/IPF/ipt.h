
#ifndef _FF_IPF_IPT_H
#define _FF_IPF_IPT_H


#include"ipf\def.h"

#include"bfc\ctc.h"

#include<memory.h>

_IPF_BEG


class pval_cast
{
public:
	template<typename _DestT, typename _SrcT>
	static _DestT cast(const _SrcT &val)
	{
		return static_cast<_DestT>(val);
	}
};

class pval_truncate
{
public:
	template<typename _DestT, typename _SrcT>
	static _DestT cast(const _SrcT &val)
	{
		const _DestT _min=NLimits<_DestT>::Min(), _max=NLimits<_DestT>::Max();

		return static_cast<_DestT>(val<_min? _min : val>_max? _max:val);
	}
};

template<typename _ValT>
struct pval_diff_type
{
	typedef int type;
};

template<>
struct pval_diff_type<float>
{
	typedef float type;
};

template<>
struct pval_diff_type<double>
{
	typedef double type;
};

template<typename _ValT>
struct pval_accum_type
{
	typedef typename pval_diff_type<_ValT>::type type;
};

template<>
struct pval_accum_type<uint>
{
	typedef uint type;
};

template<int _cn>
struct ccn
{
	enum{CNT=_cn};
public:
	operator int () const
	{
		return _cn;
	}
};

struct dcn
{
	enum{CNT=0};

	const int m_cn;
public:
	dcn(const int cn)
		:m_cn(cn)
	{
	}
	operator int () const
	{
		return m_cn;
	}
};

template<typename _Ty>
struct cnt
{
	enum{CNT=_Ty::CNT};
};

template<>
struct cnt<int>
{
	enum{CNT=0};
};

template<typename _PValT, typename _CNT>
inline void pxcpy(const _PValT *src, _PValT *dest, _CNT &cn)
{
	memcpy(dest,src,sizeof(_PValT)*cn);
}

template<typename _PValT, int CN>
inline void pxcpy(const _PValT *src, _PValT *dest, ccn<CN> &cn)
{
	memcpy(dest,src,sizeof(_PValT)*CN);
}



//=========================================================================

template<int cn>
class iop_copy
{
public:
	iop_copy(const int =0)
	{
	}
	template<typename _PValT>
	void operator()(const _PValT *pix,_PValT *pox) const
	{
		for(int ci=0;ci<cn;++ci)
			pox[ci]=pix[ci];
	}
};

#define _DEF_diop__HEAD(_class) const int m_cn; public: _class(const int cn):m_cn(cn){}

template<>
class iop_copy<0>
{
	_DEF_diop__HEAD(iop_copy)
public:
	template<typename _PValT>
	void operator()(const _PValT *pix,_PValT *pox) const
	{
		for(int ci=0;ci<m_cn;++ci)
			pox[ci]=pix[ci];
	}
};

template<int cn>
class iop_cast
{
public:
	iop_cast(int =0)
	{
	}

	template<typename _IPValT,typename _OPValT>
	void operator()(const _IPValT *pix,_OPValT *pox) const
	{
		for(int ci=0;ci<cn;++ci)
			pox[ci]=(_OPValT)pix[ci];
	}
};

template<>
class iop_cast<0>
{
	_DEF_diop__HEAD(iop_cast)
public:
	template<typename _IPValT,typename _OPValT>
	void operator()(const _IPValT *pix,_OPValT *pox) const
	{
		for(int ci=0;ci<m_cn;++ci)
			pox[ci]=(_OPValT)pix[ci];
	}
};

template<int cn, typename _ValT>
class iop_max
{
public:
	 _ValT *m_max;
public:
	iop_max(_ValT *_max, int =0)
		:m_max(_max)
	{
	}

	template<typename _IPValT>
	void operator()(const _IPValT *pix) const
	{
		for(int i=0; i<cn; ++i)
		{
			if(pix[i]>m_max[i])
				m_max[i]=_ValT(pix[i]);
		}
	}
};

template<typename _ValT>
class iop_max<0,_ValT>
{
	const int m_cn;
	 _ValT *m_max;
public:
	iop_max( _ValT *_max, int cn)
		:m_cn(cn), m_max(_max)
	{
	}
	
	template<typename _IPValT>
	void operator()(const _IPValT *pix) const
	{
		for(int i=0; i<m_cn; ++i)
		{
			if(pix[i]>m_max[i])
				m_max[i]=_ValT(pix[i]);
		}
	}
};

template<int cn, typename _ValT>
class iop_min
{
public:
	 _ValT *m_min;
public:
	iop_min(_ValT *_min,int =0)
		:m_min(_min)
	{
	}

	template<typename _IPValT>
	void operator()(const _IPValT *pix) const
	{
		for(int i=0; i<cn; ++i)
		{
			if(pix[i]<m_min[i])
				m_min[i]=_ValT(pix[i]);
		}
	}
};


template<typename _ValT>
class iop_min<0,_ValT>
{
	const int m_cn;
	 _ValT *m_min;
public:
	iop_min(_ValT *_min, int cn)
		:m_cn(cn), m_min(_min)
	{
	}
	
	template<typename _IPValT>
	void operator()(const _IPValT *pix) const
	{
		for(int i=0; i<m_cn; ++i)
		{
			if(pix[i]<m_min[i])
				m_min[i]=_ValT(pix[i]);
		}
	}
};

template<int cn>
class iop_max_p2
{//max. of corresponding pixels of two images
public:
	template<typename _IPValT0, typename _IPValT1, typename _OPValT>
	void operator()(const _IPValT0 *pix, const _IPValT1 *piy, _OPValT *po) const
	{
		for(int i=0; i<cn; ++i)
		{
			po[i]=pix[i]<piy[i]? piy[i]:pix[i];
		}
	}
};

template<int cn>
class iop_min_p2
{//min. of corresponding pixels of two images
public:
	template<typename _IPValT0, typename _IPValT1, typename _OPValT>
	void operator()(const _IPValT0 *pix, const _IPValT1 *piy, _OPValT *po) const
	{
		for(int i=0; i<cn; ++i)
		{
			po[i]=pix[i]>piy[i]? piy[i]:pix[i];
		}
	}
};


template<int cn, typename _ValT>
class iop_min_max
{
public:
	 _ValT *m_min, *m_max;
public:
	
	/*
	 _min, _max should be initialized with pixel values
	*/
	iop_min_max(_ValT *_min, _ValT *_max, int =0)
		:m_min(_min), m_max(_max)
	{
	}

	template<typename _IPValT>
	void operator()(const _IPValT *pix) const
	{
		for(int i=0; i<cn; ++i)
		{
			if(pix[i]<m_min[i])
				m_min[i]=_ValT(pix[i]);
			else
				if(pix[i]>m_max[i])
					m_max[i]=_ValT(pix[i]);
		}
	}
};


template<typename _ValT>
class iop_min_max<0,_ValT>
{
	const int m_cn;
	 _ValT *m_min, *m_max;
public:
	/*
	 _min, _max should be initialized with pixel values
	*/
	iop_min_max(_ValT *_min, _ValT *_max, int cn)
		:m_cn(cn), m_min(_min), m_max(_max)
	{
	}
	
	template<typename _IPValT>
	void operator()(const _IPValT *pix) const
	{
		for(int i=0; i<m_cn; ++i)
		{
			if(pix[i]<m_min[i])
				m_min[i]=_ValT(pix[i]);
			else
				if(pix[i]>m_max[i])
					m_max[i]=_ValT(pix[i]);
		}
	}
};


template<int cn, typename _ValT>
class iop_set
{
	const _ValT *m_pv;
public:
	iop_set(const _ValT pv[])
		:m_pv(pv)
	{
	}

	template<typename _OPValT>
	void operator()(_OPValT *pdx) const
	{
		for(int i=0; i<cn; ++i)
			pdx[i]=(_OPValT)m_pv[i];
	}
};

template<typename _ValT>
class iop_set<0,_ValT>
{
	const int m_cn;
	const _ValT *m_pv;
public:
	iop_set(const _ValT pv[], int cn)
		:m_pv(pv), m_cn(cn)
	{
	}

	template<typename _OPValT>
	void operator()(_OPValT *pdx) const
	{
		for(int i=0; i<m_cn; ++i)
			pdx[i]=(_OPValT)m_pv[i];
	}
};


template<int cn,typename _MapValT>
class iop_map
{
	const _MapValT *m_pmap;
public:
	iop_map(const _MapValT *pmap,int =0)
		:m_pmap(pmap)
	{
	}
	template<typename _IPValT,typename _OPValT>
	void operator()(const _IPValT *pix,_OPValT *pox) const
	{
		for(int ci=0;ci<cn;++ci)
		//	pox[ci]=(_OPValT)m_pmap[pix[ci]];
			memcpy(pox,&m_pmap[pix[ci]],sizeof(_MapValT));
	}
};

template<typename _MapValT>
class iop_map<0,_MapValT>
{
	const int       m_cn;
	const _MapValT *m_pmap;
public:
	iop_map(const _MapValT *pmap, int cn)
		:m_cn(cn),m_pmap(pmap)
	{
	}
	template<typename _IPValT,typename _OPValT>
	void operator()(const _IPValT *pix,_OPValT *pox) const
	{
		for(int ci=0;ci<m_cn;++ci)
			pox[ci]=(_OPValT)m_pmap[pix[ci]];
	}
};


template<int cn,typename _ScaleValT, typename _CastT=pval_cast>
class iop_scale
{
	const _ScaleValT m_scale;
public:
	iop_scale(const _ScaleValT scale, int =0)
		:m_scale(scale)
	{
	}
	template<typename _IPValT,typename _OPValT>
	void operator()(const _IPValT *pix,_OPValT *pox) const
	{
		for(int ci=0;ci<cn;++ci)
			pox[ci]=_CastT::cast<_OPValT>(pix[ci]*m_scale);
	}
};

template<typename _ScaleValT, typename _CastT>
class iop_scale<0,_ScaleValT,_CastT>
{
	const int		 m_cn;
	const _ScaleValT m_scale;
public:
	iop_scale(const _ScaleValT scale, int cn)
		:m_cn(cn),m_scale(scale)
	{
	}
	template<typename _IPValT,typename _OPValT>
	void operator()(const _IPValT *pix,_OPValT *pox) const
	{
		for(int ci=0;ci<m_cn;++ci)
			pox[ci]=_CastT::cast<_OPValT>(pix[ci]*m_scale);
	}
};
//
//template<int cn,typename _ConstValT, typename _CastT=pval_cast>
//class iop_add_const
//{
//	const _ConstValT m_acval;
//public:
//	iop_add_const(const _ConstValT acval)
//		:m_acval(acval)
//	{
//	}
//	template<typename _IPValT,typename _OPValT>
//	void operator()(const _IPValT *pix,_OPValT *pox) const
//	{
//		for(int ci=0;ci<cn;++ci)
//			pox[ci]=_CastT::cast<_OPValT>(pix[ci]+m_acval);
//	}
//};
//
//template<typename _ConstValT, typename _CastT=pval_cast>
//class diop_add_const
//{
//	const int		 m_cn;
//	const _ConstValT m_acval;
//public:
//	diop_add_const(const int cn,const _ConstValT acval)
//		:m_cn(cn),m_acval(acval)
//	{
//	}
//	template<typename _IPValT,typename _OPValT>
//	void operator()(const _IPValT *pix,_OPValT *pox) const
//	{
//		for(int ci=0;ci<m_cn;++ci)
//			pox[ci]=_CastT<_OPValT>(pix[ci]+m_acval);
//	}
//};
//
//template<int cn,typename _TValT, typename _CastT=pval_cast>
//class iop_affine
//{
//	const _TValT m_scale,m_shift;
//public:
//	iop_affine(const _TValT scale,const _TValT shift)
//		:m_scale(scale),m_shift(shift)
//	{
//	}
//	template<typename _IPValT,typename _OPValT>
//	void operator()(const _IPValT *pix,_OPValT *pox) const
//	{
//		for(int ci=0;ci<cn;++ci)
//			pox[ci]=_CastT<_OPValT>(pix[ci]*m_scale+m_shift);
//	}
//};
//
//template<typename _TValT, typename _CastT=pval_cast>
//class diop_affine
//{
//	const int	 m_cn;
//	const _TValT m_scale,m_shift;
//public:
//	diop_affine(const int cn,const _TValT scale,const _TValT shift)
//		:m_cn(cn),m_scale(scale),m_shift(shift)
//	{
//	}
//	template<typename _IPValT,typename _OPValT>
//	void operator()(const _IPValT *pix,_OPValT *pox) const
//	{
//		for(int ci=0;ci<m_cn;++ci)
//			pox[ci]=_CastT<_OPValT>(pix[ci]*m_scale+m_shift);
//	}
//};
//
//template<int cn,typename _TValT,typename _OValT>
//class iop_threshold
//{
//	const _TValT m_threshold;
//	const _OValT m_val0,m_val1;
//public:
//	iop_threshold(const _TValT threshold,const _OValT val0,const _OValT val1)
//		:m_threshold(threshold),m_val0(val0),m_val1(val1)
//	{
//	}
//	template<typename _IPValT,typename _OPValT>
//	void operator()(const _IPValT *pix,_OPValT *pox) const
//	{
//		for(int ci=0;ci<cn;++ci)
//			pox[ci]=(_OPValT)(pix[ci]<m_threshold? m_val0:m_val1);
//	}
//};
//
//template<typename _TValT,typename _OValT>
//class diop_threshold
//{
//	const int	 m_cn;
//	const _TValT m_threshold;
//	const _OValT m_val0,m_val1;
//public:
//	diop_threshold(const int cn,const _TValT threshold,const _OValT val0,const _OValT val1)
//		:m_cn(cn),m_threshold(threshold),m_val0(val0),m_val1(val1)
//	{
//	}
//	template<typename _IPValT,typename _OPValT>
//	void operator()(const _IPValT *pix,_OPValT *pox) const
//	{
//		for(int ci=0;ci<m_cn;++ci)
//			pox[ci]=(_OPValT)(pix[ci]<m_threshold? m_val0:m_val1);
//	}
//};
//
//template<int cn>
//class iop_add
//{
//public:
//	template<typename _IPValT0,typename _IPValT1,typename _OPValT>
//	void operator()(const _IPValT0 *pix0,const _IPValT1 *pix1,_OPValT *pox)
//	{
//		for(int ci=0;ci<cn;++ci)
//			pox[ci]=(_OPValT)(pix0[ci]+pix1[ci]);
//	}
//};
//
//class diop_add
//{
//	const int m_cn;
//public:
//	diop_add(const int cn)
//		:m_cn(cn)
//	{
//	}
//	template<typename _IPValT0,typename _IPValT1,typename _OPValT>
//	void operator()(const _IPValT0 *pix0,const _IPValT1 *pix1,_OPValT *pox)
//	{
//		for(int ci=0;ci<m_cn;++ci)
//			pox[ci]=(_OPValT)(pix0[ci]+pix1[ci]);
//	}
//};
//
template<int cn>
class iop_sub
{
public:
	iop_sub(int =0)
	{
	}

	template<typename _IPValT0,typename _IPValT1,typename _OPValT>
	void operator()(const _IPValT0 *pix0,const _IPValT1 *pix1,_OPValT *pox)
	{
		for(int ci=0;ci<cn;++ci)
			pox[ci]=(_OPValT)(pix0[ci]-pix1[ci]);
	}
};

template<>
class iop_sub<0>
{
	const int m_cn;
public:
	iop_sub(const int cn)
		:m_cn(cn)
	{
	}
	template<typename _IPValT0,typename _IPValT1,typename _OPValT>
	void operator()(const _IPValT0 *pix0,const _IPValT1 *pix1,_OPValT *pox)
	{
		for(int ci=0;ci<m_cn;++ci)
			pox[ci]=(_OPValT)(pix0[ci]-pix1[ci]);
	}
};

//template<int cn>
//class iop_mul
//{
//public:
//	template<typename _IPValT0,typename _IPValT1,typename _OPValT>
//	void operator()(const _IPValT0 *pix0,const _IPValT1 *pix1,_OPValT *pox)
//	{
//		for(int ci=0;ci<cn;++ci)
//			pox[ci]=(_OPValT)(pix0[ci]*pix1[ci]);
//	}
//};
//
//class diop_mul
//{
//	const int m_cn;
//public:
//	diop_mul(const int cn)
//		:m_cn(cn)
//	{
//	}
//	template<typename _IPValT0,typename _IPValT1,typename _OPValT>
//	void operator()(const _IPValT0 *pix0,const _IPValT1 *pix1,_OPValT *pox)
//	{
//		for(int ci=0;ci<m_cn;++ci)
//			pox[ci]=(_OPValT)(pix0[ci]*pix1[ci]);
//	}
//};
//
//template<int cn>
//class iop_div
//{
//public:
//	template<typename _IPValT0,typename _IPValT1,typename _OPValT>
//	void operator()(const _IPValT0 *pix0,const _IPValT1 *pix1,_OPValT *pox)
//	{
//		for(int ci=0;ci<cn;++ci)
//			pox[ci]=(_OPValT)(pix0[ci]/pix1[ci]);
//	}
//};
//
//class diop_div
//{
//	const int m_cn;
//public:
//	diop_div(const int cn)
//		:m_cn(cn)
//	{
//	}
//	template<typename _IPValT0,typename _IPValT1,typename _OPValT>
//	void operator()(const _IPValT0 *pix0,const _IPValT1 *pix1,_OPValT *pox)
//	{
//		for(int ci=0;ci<m_cn;++ci)
//			pox[ci]=(_OPValT)(pix0[ci]/pix1[ci]);
//	}
//};
//
//template<int cn>
//class iop_or
//{
//public:
//	template<typename _IPValT0,typename _IPValT1,typename _OPValT>
//	void operator()(const _IPValT0 *pix0,const _IPValT1 *pix1,_OPValT *pox)
//	{
//		for(int ci=0;ci<cn;++ci)
//			pox[ci]=(_OPValT)(pix0[ci]|pix1[ci]);
//	}
//};
//
//class diop_or
//{
//	const int m_cn;
//public:
//	diop_or(const int cn)
//		:m_cn(cn)
//	{
//	}
//	template<typename _IPValT0,typename _IPValT1,typename _OPValT>
//	void operator()(const _IPValT0 *pix0,const _IPValT1 *pix1,_OPValT *pox)
//	{
//		for(int ci=0;ci<m_cn;++ci)
//			pox[ci]=(_OPValT)(pix0[ci]|pix1[ci]);
//	}
//};
//
template<int cn>
class iop_and
{
public:
	template<typename _IPValT0,typename _IPValT1,typename _OPValT>
	void operator()(const _IPValT0 *pix0,const _IPValT1 *pix1,_OPValT *pox)
	{
		for(int ci=0;ci<cn;++ci)
			pox[ci]=(_OPValT)(pix0[ci]&pix1[ci]);
	}
};
//
//class diop_and
//{
//	const int m_cn;
//public:
//	diop_and(const int cn)
//		:m_cn(cn)
//	{
//	}
//	template<typename _IPValT0,typename _IPValT1,typename _OPValT>
//	void operator()(const _IPValT0 *pix0,const _IPValT1 *pix1,_OPValT *pox)
//	{
//		for(int ci=0;ci<m_cn;++ci)
//			pox[ci]=(_OPValT)(pix0[ci]&pix1[ci]);
//	}
//};
//
//template<int cn>
//class iop_xor
//{
//public:
//	template<typename _IPValT0,typename _IPValT1,typename _OPValT>
//	void operator()(const _IPValT0 *pix0,const _IPValT1 *pix1,_OPValT *pox)
//	{
//		for(int ci=0;ci<cn;++ci)
//			pox[ci]=(_OPValT)(pix0[ci]^pix1[ci]);
//	}
//};
//
//class diop_xor
//{
//	const int m_cn;
//public:
//	diop_xor(const int cn)
//		:m_cn(cn)
//	{
//	}
//	template<typename _IPValT0,typename _IPValT1,typename _OPValT>
//	void operator()(const _IPValT0 *pix0,const _IPValT1 *pix1,_OPValT *pox)
//	{
//		for(int ci=0;ci<m_cn;++ci)
//			pox[ci]=(_OPValT)(pix0[ci]^pix1[ci]);
//	}
//};
//
//template<int cn>
//class iop_not
//{
//public:
//	template<typename _IPValT0,typename _OPValT>
//	void operator()(const _IPValT0 *pix0,_OPValT *pox)
//	{
//		for(int ci=0;ci<cn;++ci)
//			pox[ci]=(_OPValT)(~pix0[ci]);
//	}
//};
//
//class diop_not
//{
//	const int m_cn;
//public:
//	diop_not(const int cn)
//		:m_cn(cn)
//	{
//	}
//	template<typename _IPValT0,typename _OPValT>
//	void operator()(const _IPValT0 *pix0,_OPValT *pox)
//	{
//		for(int ci=0;ci<m_cn;++ci)
//			pox[ci]=(_OPValT)(~pix0[ci]);
//	}
//};
//
//template<int cn>
//class iop_lshift
//{
//	const int m_shift;
//public:
//	iop_lshift(const int shift)
//		:m_shift(shift)
//	{
//	}
//	template<typename _IPValT0,typename _OPValT>
//	void operator()(const _IPValT0 *pix0,_OPValT *pox)
//	{
//		for(int ci=0;ci<cn;++ci)
//			pox[ci]=(_OPValT)(pix0[ci]<<m_shift);
//	}
//};
//
//class diop_lshift
//{
//	const int m_cn,m_shift;
//public:
//	diop_lshift(const int cn,const int shift)
//		:m_cn(cn),m_shift(shift)
//	{
//	}
//	template<typename _IPValT0,typename _OPValT>
//	void operator()(const _IPValT0 *pix0,_OPValT *pox)
//	{
//		for(int ci=0;ci<m_cn;++ci)
//			pox[ci]=(_OPValT)(pix0[ci]<<m_shift);
//	}
//};
//
//template<int cn>
//class iop_rshift
//{
//	const int m_shift;
//public:
//	iop_rshift(const int shift)
//		:m_shift(shift)
//	{
//	}
//	template<typename _IPValT0,typename _OPValT>
//	void operator()(const _IPValT0 *pix0,_OPValT *pox)
//	{
//		for(int ci=0;ci<cn;++ci)
//			pox[ci]=(_OPValT)(pix0[ci]>>m_shift);
//	}
//};
//
//class diop_rshift
//{
//	const int m_cn,m_shift;
//public:
//	diop_rshift(const int cn,const int shift)
//		:m_cn(cn),m_shift(shift)
//	{
//	}
//	template<typename _IPValT0,typename _OPValT>
//	void operator()(const _IPValT0 *pix0,_OPValT *pox)
//	{
//		for(int ci=0;ci<m_cn;++ci)
//			pox[ci]=(_OPValT)(pix0[ci]>>m_shift);
//	}
//};


template<int cn>
class iop_alpha_blend_i8u //F,B should be int, Alpha should be in the range of [0,255]
{
public:
	iop_alpha_blend_i8u(int =0)
	{
	}

	template<typename _IPValT0,typename _IPValT1, typename _IPValT2, typename _OPValT>
	void operator()(const _IPValT0 *f, const _IPValT1 *a, const _IPValT2 *b, _OPValT *c)
	{
		for(int ci=0;ci<cn;++ci)
			c[ci]=(_OPValT)( ((int(b[ci])<<8)+(int(f[ci])-b[ci])*a[0])>>8 );
	}
};

template<>
class iop_alpha_blend_i8u<0>
{
	const int m_cn;
public:
	iop_alpha_blend_i8u(const int cn)
		:m_cn(cn)
	{
	}
	template<typename _IPValT0,typename _IPValT1, typename _IPValT2, typename _OPValT>
	void operator()(const _IPValT0 *f, const _IPValT1 *a, const _IPValT2 *b, _OPValT *c)
	{
		for(int ci=0;ci<m_cn;++ci)
			c[ci]=(_OPValT)( ((int(b[ci])<<8)+(int(f[ci])-b[ci])*a[0])>>8 );
	}
};

template<int cn>
class iop_alpha_blend_ff //F,B can be float, Alpha should be in the range of [0,1]
{
public:
	iop_alpha_blend_ff(int =0)
	{
	}

	template<typename _IPValT0,typename _IPValT1, typename _IPValT2, typename _OPValT>
	void operator()(const _IPValT0 *f, const _IPValT1 *a, const _IPValT2 *b, _OPValT *c)
	{
		for(int ci=0;ci<cn;++ci)
		//	c[ci]=(_OPValT)( (f[ci]*a[ci]+b[ci]*(1.0-a[ci])) );
			c[ci]=(_OPValT)( b[ci]+(f[ci]-b[ci])*a[0] );
	}
};

template<>
class iop_alpha_blend_ff<0>
{
	const int m_cn;
public:
	iop_alpha_blend_ff(const int cn)
		:m_cn(cn)
	{
	}
	
	template<typename _IPValT0,typename _IPValT1, typename _IPValT2, typename _OPValT>
	void operator()(const _IPValT0 *f, const _IPValT1 *a, const _IPValT2 *b, _OPValT *c)
	{
		for(int ci=0;ci<m_cn;++ci)
			c[ci]=(_OPValT)( b[ci]+(f[ci]-b[ci])*a[0] );
	}
};

//===============================================================================

template<typename _IOPT,typename _IPValT0>
class iop_bind_i0
{
	const _IPValT0  *m_psx0;
	_IOPT			 &m_op;
public:
	iop_bind_i0(const _IPValT0 *psx0,_IOPT &op)
		:m_psx0(psx0),m_op(op)
	{
	}
	template<typename _OPValT>
	void operator()(_OPValT *pox) const
	{
		m_op(m_psx0,pox);
	}
	template<typename _IPValT1,typename _OPValT>
	void operator()(const _IPValT1 *pix1,_OPValT *pox) const
	{
		m_op(m_psx0,pix1,pox);
	}
	template<typename _IPValT1,typename _IPValT2,typename _OPValT>
	void operator()(const _IPValT1 *pix1,const _IPValT2 *pix2,_OPValT *pox) const
	{
		m_op(m_psx0,pix1,pix2,pox);
	}
};

template<typename _IOPT,typename _IPValT0>
inline iop_bind_i0<_IOPT,_IPValT0> bind_i0(const _IPValT0 *psx0,_IOPT &op)
{
	return iop_bind_i0<_IOPT,_IPValT0>(psx0,op);
}

template<typename _IOPT,typename _IPValT1>
class iop_bind_i1
{
	const _IPValT1  *m_psx1;
	_IOPT			 &m_op;
public:
	iop_bind_i1(const _IPValT1 *psx1,_IOPT &op)
		:m_psx1(psx1),m_op(op)
	{
	}
	template<typename _IPValT0,typename _OPValT>
	void operator()(const _IPValT0 *pix0,_OPValT *pox) const
	{
		m_op(pix0,m_psx1,pox);
	}
	template<typename _IPValT0,typename _IPValT2,typename _OPValT>
	void operator()(const _IPValT0 *pix0,const _IPValT2 *pix2,_OPValT *pox) const
	{
		m_op(pix0,m_psx1,pix2,pox);
	}
};

template<typename _IOPT,typename _IPValT1>
inline iop_bind_i1<_IOPT,_IPValT1> bind_i1(const _IPValT1 *psx1,_IOPT &op)
{
	return iop_bind_i1<_IOPT,_IPValT1>(psx1,op);
}

template<typename _IOPT,typename _IPValT2>
class iop_bind_i2
{
	const _IPValT2  *m_psx2;
	_IOPT			 &m_op;
public:
	iop_bind_i2(const _IPValT2 *psx2,_IOPT &op)
		:m_psx2(psx2),m_op(op)
	{
	}
	template<typename _IPValT0,typename _IPValT1,typename _OPValT>
	void operator()(const _IPValT0 *pix0,const _IPValT1 *pix1,_OPValT *pox) const
	{
		m_op(pix0,pix1,m_psx2,pox);
	}
};

template<typename _IOPT,typename _IPValT2>
inline iop_bind_i2<_IOPT,_IPValT2> bind_i2(const _IPValT2 *psx2,_IOPT &op)
{
	return iop_bind_i2<_IOPT,_IPValT2>(psx2,op);
}

template<typename _IOPT>
class iop_inplace_i0
{
	_IOPT			 &m_op;
public:
	iop_inplace_i0(_IOPT &op)
		:m_op(op)
	{
	}
	template<typename _OPValT>
	void operator()(_OPValT *pox) const
	{
		m_op(pox,pox);
	}
	template<typename _IPValT1,typename _OPValT>
	void operator()(const _IPValT1 *pix1,_OPValT *pox) const
	{
		m_op(pox,pix1,pox);
	}
	template<typename _IPValT1,typename _IPValT2,typename _OPValT>
	void operator()(const _IPValT1 *pix1,const _IPValT2 *pix2,_OPValT *pox) const
	{
		m_op(pox,pix1,pix2,pox);
	}
};

template<typename _IOPT>
inline iop_inplace_i0<_IOPT> inplace_i0(_IOPT &op)
{
	return iop_inplace_i0<_IOPT>(op);
}

template<typename _IOPT>
class iop_inplace_i1
{
	_IOPT			 &m_op;
public:
	iop_inplace_i1(_IOPT &op)
		:m_op(op)
	{
	}
	template<typename _IPValT0,typename _OPValT>
	void operator()(const _IPValT0 *pix0,_OPValT *pox) const
	{
		m_op(pix0,pox,pox);
	}
	template<typename _IPValT0,typename _IPValT2,typename _OPValT>
	void operator()(const _IPValT0 *pix0,const _IPValT2 *pix2,_OPValT *pox) const
	{
		m_op(pix0,pox,pix2,pox);
	}
};

template<typename _IOPT>
inline iop_inplace_i1<_IOPT> inplace_i1(_IOPT &op)
{
	return iop_inplace_i1<_IOPT>(op);
}

template<typename _IOPT>
class iop_inplace_i2
{
	_IOPT			 &m_op;
public:
	iop_inplace_i2(_IOPT &op)
		:m_op(op)
	{
	}
	template<typename _IPValT0,typename _IPValT1,typename _OPValT>
	void operator()(const _IPValT0 *pix0,const _IPValT1 *pix1,_OPValT *pox) const
	{
		m_op(pix0,pix1,pox,pox);
	}
};

template<typename _IOPT>
inline iop_inplace_i2<_IOPT> inplace_i2(_IOPT &op)
{
	return iop_inplace_i2<_IOPT>(op);
}


#undef _DEF_diop__HEAD

//======================================================================================

template<typename _MaskValT>
class iop_mask_zero
{
public:
	bool operator()(_MaskValT mvx) const
	{
		return mvx==0;
	}
};


template<typename _MaskValT>
class iop_mask_nzero //not-zero
{
public:
	
	bool operator()(_MaskValT mvx) const
	{
		return mvx!=0;
	}
};

template<typename _MaskValT>
class iop_mask_ge
{
	const _MaskValT m_mval;
public:
	iop_mask_ge(const _MaskValT mval)
		:m_mval(mval)
	{
	}
	bool operator()(_MaskValT mvx) const
	{
		return mvx>=m_mval;
	}
};

template<typename _MaskValT>
class iop_mask_eq
{
	const _MaskValT m_mval;
public:
	iop_mask_eq(const _MaskValT mval)
		:m_mval(mval)
	{
	}
	bool operator()(_MaskValT mvx) const
	{
		return mvx==m_mval;
	}
};
template<typename _MaskValT>
class iop_mask_le
{
	const _MaskValT m_mval;
public:
	iop_mask_le(const _MaskValT mval)
		:m_mval(mval)
	{
	}
	bool operator()(_MaskValT mvx) const
	{
		return mvx<=m_mval;
	}
};

template<typename _MaskValT>
class iop_mask_neq
{
	const _MaskValT m_mval;
public:
	iop_mask_neq(const _MaskValT mval)
		:m_mval(mval)
	{
	}
	bool operator()(_MaskValT mvx) const
	{
		return mvx!=m_mval;
	}
};

template<typename _MaskValT>
class iop_mask_rng //In-Range
{
	const _MaskValT m_mvbeg,m_mvend;
public:
	iop_mask_rng(const _MaskValT mvbeg,const _MaskValT mvend)
		:m_mvbeg(mvbeg),m_mvend(mvend)
	{
	}
	bool operator()(_MaskValT mvx) const
	{
		return mvx>=m_mvbeg&&mvx<m_mvend;
	}
};

template<typename _MaskValT>
class iop_mask_nrng //Not-In-Range
{
	const _MaskValT m_mvbeg,m_mvend;
public:
	iop_mask_nrng(const _MaskValT mvbeg,const _MaskValT mvend)
		:m_mvbeg(mvbeg),m_mvend(mvend)
	{
	}
	bool operator()(_MaskValT mvx) const
	{
		return mvx<m_mvbeg||mvx>=m_mvend;
	}
};

//==================================================================================================

template<typename _SrcValT,typename _OpT>
inline _OpT& for_each_pixel_1_0(const _SrcValT *pSrc,const int width,const int height,const int istride,const int icn,  _OpT &op)
{
	if(pSrc)
	{
		for(int yi=0;yi<height;++yi,pSrc+=istride)
		{
			const _SrcValT *pix=pSrc;

			for(int xi=0;xi<width;++xi,pix+=icn)
			{
				op(pix);
			}
		}
	}

	return op;
}

template<typename _SrcValT,typename _OpT>
inline _OpT& for_boundary_pixel_1_0(const _SrcValT *pSrc,const int width,const int height,const int istride,const int icn,
									_OpT &op,
									const int nTopLine,const int nBottomLine,const int nLeftPixel,const int nRightPixel
								)
{
	if(pSrc)
	{
		for_each_pixel_1_0(pSrc,width,nTopLine,istride,icn,op);

		for_each_pixel_1_0(pSrc+nTopLine*istride,nLeftPixel,height-nTopLine-nBottomLine,istride,icn, op);

		for_each_pixel_1_0(pSrc+nTopLine*istride+(width-nRightPixel)*icn,nRightPixel,height-nTopLine-nBottomLine,istride,icn, op);

		for_each_pixel_1_0(pSrc+(height-nBottomLine)*istride,width,nBottomLine,istride,icn, op);
	}

	return op;
}

template<typename _SrcValT,typename _MaskValT, typename _OpT,typename _MaskOpT>
inline _OpT& for_masked_pixel_1_0(const _SrcValT *pSrc,const int width,const int height,const int istride,const int icn,
								 _OpT &op,
								 const _MaskValT *pMask,const int mstride,
								 _MaskOpT &maskOP
							  )
{
	if(pSrc && pMask)
	{
		for(int yi=0;yi<height;++yi,pSrc+=istride,pMask+=mstride)
		{
			const _SrcValT *pdx=pSrc;

			for(int xi=0;xi<width;++xi,pdx+=icn)
			{
				if(maskOP(pMask[xi]))
					op(pdx);
			}
		}
	}

	return op;
}

template<typename _DestValT,typename _OpT>
inline _OpT& for_each_pixel_0_1(_DestValT *pDest,const int width,const int height,const int ostride,const int ocn, _OpT &op)
{
	if(pDest)
	{
		for(int yi=0;yi<height;++yi,pDest+=ostride)
		{
			_DestValT *pdx=pDest;

			for(int xi=0;xi<width;++xi,pdx+=ocn)
			{
				op(pdx);
			}
		}
	}

	return op;
}

template<typename _DestValT,typename _OpT>
inline _OpT& for_boundary_pixel_0_1(_DestValT *pDest,const int width,const int height,const int ostride,const int ocn,
									_OpT &op,
									const int nTopLine,const int nBottomLine,const int nLeftPixel,const int nRightPixel
								)
{
	if(pDest)
	{
		for_each_pixel_0_1(pDest,width,nTopLine,ostride,ocn,op);

		for_each_pixel_0_1(pDest+nTopLine*ostride,nLeftPixel,height-nTopLine-nBottomLine,ostride,ocn, op);

		for_each_pixel_0_1(pDest+nTopLine*ostride+(width-nRightPixel)*ocn,nRightPixel,height-nTopLine-nBottomLine,ostride,ocn, op);

		for_each_pixel_0_1(pDest+(height-nBottomLine)*ostride,width,nBottomLine,ostride,ocn, op);
	}

	return op;
}

template<typename _DestValT,typename _MaskValT, typename _OpT,typename _MaskOpT>
inline _OpT& for_masked_pixel_0_1(_DestValT *pDest,const int width,const int height,const int ostride,const int ocn,
								 _OpT &op,
								 const _MaskValT *pMask,const int mstride,
								 _MaskOpT &maskOP
							  )
{
	if(pDest && pMask)
	{
		for(int yi=0;yi<height;++yi,pDest+=ostride,pMask+=mstride)
		{
			_DestValT *pdx=pDest;

			for(int xi=0;xi<width;++xi,pdx+=ocn)
			{
				if(maskOP(pMask[xi]))
					op(pdx);
			}
		}
	}

	return op;
}

template<typename _SrcValT,typename _DestValT,typename _OpT>
inline _OpT& for_each_pixel_1_1(const _SrcValT *pSrc,const int width,const int height,const int istride,const int icn,
								_DestValT *pDest,const int ostride,const int ocn,
								_OpT &op
								)
{
	if(pSrc && pDest)
	{
		for(int yi=0;yi<height;++yi,pSrc+=istride, pDest+=ostride)
		{
			const _SrcValT *pix=pSrc;
			_DestValT *pdx=pDest;

			for(int xi=0;xi<width;++xi,pix+=icn,pdx+=ocn)
			{
				op(pix,pdx);
			}
		}
	}

	return op;
}

template<typename _SrcValT,typename _DestValT,typename _OpT>
inline _OpT& for_boundary_pixel_1_1(const _SrcValT *pSrc,const int width,const int height,const int istride,const int icn,
								_DestValT *pDest,const int ostride,const int ocn,
								_OpT &op,
								const int nTopLine,const int nBottomLine,const int nLeftPixel,const int nRightPixel
								)
{
	if(pSrc && pDest)
	{
		for_each_pixel_1_1(pSrc,width,nTopLine,istride,icn,pDest,ostride,ocn,op);

		for_each_pixel_1_1(pSrc+nTopLine*istride,nLeftPixel,height-nTopLine-nBottomLine,istride,icn,pDest+nTopLine*ostride,ostride,ocn, op);

		for_each_pixel_1_1(pSrc+nTopLine*istride+(width-nRightPixel)*icn,nRightPixel,height-nTopLine-nBottomLine,istride,icn,
							pDest+nTopLine*ostride+(width-nRightPixel)*ocn,ostride,ocn,op);

		for_each_pixel_1_1(pSrc+(height-nBottomLine)*istride,width,nBottomLine,istride,icn,
						pDest+(height-nBottomLine)*ostride,ostride,ocn, op);
	}

	return op;
}

template<typename _SrcValT,typename _DestValT,typename _MaskValT,typename _OpT,typename _MaskOpT>
inline _OpT& for_masked_pixel_1_1(const _SrcValT *pSrc,const int width,const int height,const int istride,const int icn,
								_DestValT *pDest,const int ostride,const int ocn,
								_OpT &op,
								const _MaskValT *pMask,const int mstride,
								_MaskOpT &maskOp
								)
{
	if(pSrc && pDest && pMask)
	{
		for(int yi=0;yi<height;++yi,pSrc+=istride,pDest+=ostride,pMask+=mstride)
		{
			const _SrcValT *pix=pSrc;
			_DestValT *pdx=pDest;

			for(int xi=0;xi<width;++xi,pix+=icn,pdx+=ocn)
			{
				if(maskOp(pMask[xi]))
					op(pix,pdx);
			}
		}
	}

	return op;
}

template<typename _SrcValT0,typename _SrcValT1,typename _DestValT,typename _OpT>
inline _OpT& for_each_pixel_2_1(const _SrcValT0 *pSrc0,const int width,const int height,const int istride0,const int icn0,
							 const _SrcValT1 *pSrc1,const int istride1,const int icn1,
							_DestValT *pDest,const int ostride,const int ocn,
							_OpT &op
							)
{
	if(pSrc0 && pSrc1 && pDest)
	{
		for(int yi=0;yi<height;++yi,pSrc0+=istride0,pSrc1+=istride1,pDest+=ostride)
		{
			const _SrcValT0 *pix0=pSrc0;
			const _SrcValT1 *pix1=pSrc1;
			_DestValT *pdx=pDest;

			for(int xi=0;xi<width;++xi,pix0+=icn0,pix1+=icn1,pdx+=ocn)
			{
				op(pix0,pix1,pdx);
			}
		}
	}

	return op;
}

template<typename _SrcValT0,typename _SrcValT1,typename _DestValT,typename _OpT>
inline _OpT& for_boundary_pixel_2_1(const _SrcValT0 *pSrc0,const int width,const int height,const int istride0,const int icn0,
							 const _SrcValT1 *pSrc1,const int istride1,const int icn1,
							 _DestValT *pDest,const int ostride,const int ocn,
							 _OpT &op,
							 const int nTopLine,const int nBottomLine,const int nLeftPixel,const int nRightPixel
							)
{
	if(pSrc0 && pSrc1 && pDest)
	{
		for_each_pixel_2_1(pSrc0,width,nTopLine,istride0,icn0,pSrc1,istride1,icn1,pDest,ostride,ocn,op);

		for_each_pixel_2_1(pSrc0+nTopLine*istride0,nLeftPixel,height-nTopLine-nBottomLine,istride0,icn0,
						 pSrc1+nTopLine*istride1,istride1,icn1,
						 pDest,nTopLine*ostride,ostride,ocn,
						 op);

		for_each_pixel_2_1(pSrc0+nTopLine*istride0+(width-nRightPixel)*icn0,nRightPixel,height-nTopLine-nBottomLine,istride0,icn0,
						 pSrc1+nTopLine*istride1+(width-nRightPixel)*icn1,istride1,icn1,
						 pDest+nTopLine*ostride+(width-nRightPixel)*ocn,ostride,ocn,
						 op);

		for_each_pixel_2_1(pSrc0+(height-nBottomLine)*istride0,width,nBottomLine,istride0,icn0,
						 pSrc1+(height-nBottomLine)*istride1,istride1,icn1,
						 pDest+(height-nBottomLine)*ostride,ostride,ocn,
						 op);
	}

	return op;
}

template<typename _SrcValT0,typename _SrcValT1,typename _DestValT,typename _MaskValT,typename _OpT,typename _MaskOpT>
inline _OpT& for_masked_pixel_2_1(const _SrcValT0 *pSrc0,const int width,const int height,const int istride0,const int icn0,
							 const _SrcValT1 *pSrc1,const int istride1,const int icn1,
							_DestValT *pDest,const int ostride,const int ocn,
							_OpT &op,
							const _MaskValT *pMask,const int mstride,
							_MaskOpT &maskOp
							)
{
	if(pSrc0 && pSrc1 && pDest && pMask)
	{
		for(int yi=0;yi<height;++yi,pSrc0+=istride0,pSrc+=istride1,pDest+=ostride,pMask+=mstride)
		{
			const _SrcValT0 *pix0=pSrc0;
			const _SrcValT1 *pix1=pSrc1;
			_DestValT *pdx=pDest;

			for(int xi=0;xi<width;++xi,pix0+=icn0,pix1+=icn1,pdx+=ocn)
			{
				if(maskOp(pMask[xi]))
					op(pix0,pix1,pdx);
			}
		}
	}

	return op;
}


template<typename _SrcValT0,typename _SrcValT1,typename _SrcValT2,typename _DestValT,typename _OpT>
inline _OpT& for_each_pixel_3_1(const _SrcValT0 *pSrc0,const int width,const int height,const int istride0,const int icn0,
							 const _SrcValT1 *pSrc1,const int istride1,const int icn1,
							 const _SrcValT2 *pSrc2,const int istride2,const int icn2,
							_DestValT *pDest,const int ostride,const int ocn,
							_OpT &op
							)
{
	if(pSrc0 && pSrc1 && pSrc2 && pDest)
	{
		for(int yi=0;yi<height;++yi,pSrc0+=istride0,pSrc1+=istride1,pSrc2+=istride2,pDest+=ostride)
		{
			const _SrcValT0 *pix0=pSrc0;
			const _SrcValT1 *pix1=pSrc1;
			const _SrcValT2 *pix2=pSrc2;
			_DestValT *pdx=pDest;

			for(int xi=0;xi<width;++xi,pix0+=icn0,pix1+=icn1,pix2+=icn2,pdx+=ocn)
			{
				op(pix0,pix1,pix2,pdx);
			}
		}
	}

	return op;
}

template<typename _SrcValT0,typename _SrcValT1,typename _SrcValT2,typename _DestValT,typename _OpT>
inline _OpT& for_boundary_pixel_3_1(const _SrcValT0 *pSrc0,const int width,const int height,const int istride0,const int icn0,
							 const _SrcValT1 *pSrc1,const int istride1,const int icn1,
							 const _SrcValT2 *pSrc2,const int istride2,const int icn2,
							 _DestValT *pDest,const int ostride,const int ocn,
							 _OpT &op,
							 const int nTopLine,const int nBottomLine,const int nLeftPixel,const int nRightPixel
							)
{
	if(pSrc0 && pSrc1 && pSrc2 && pDest)
	{
		for_each_pixel_3_1(pSrc0,width,nTopLine,istride0,icn0,pSrc1,istride1,icn1,pSrc2,istride2,icn2,pDest,ostride,ocn,op);

		for_each_pixel_3_1(pSrc0+nTopLine*istride0,nLeftPixel,height-nTopLine-nBottomLine,istride0,icn0,
						 pSrc1+nTopLine*istride1,istride1,icn1,
						 pSrc2+nTopLine*istride2,istride2,icn2,
						 pDest+nTopLine*ostride,ostride,ocn,
						 op);

		for_each_pixel_3_1(pSrc0+nTopLine*istride0+(width-nRightPixel)*icn0,nRightPixel,height-nTopLine-nBottomLine,istride0,icn0,
						 pSrc1+nTopLine*istride1+(width-nRightPixel)*icn1,istride1,icn1,
						 pSrc2+nTopLine*istride2+(width-nRightPixel)*icn2,istride2,icn2,
						 pDest+nTopLine*ostride+(width-nRightPixel)*ocn,ostride,ocn,
						 op);

		for_each_pixel_3_1(pSrc0+(height-nBottomLine)*istride0,width,nBottomLine,istride0,icn0,
						 pSrc1+(height-nBottomLine)*istride1,istride1,icn1,
						 pSrc2+(height-nBottomLine)*istride2,istride2,icn2,
						 pDest+(height-nBottomLine)*ostride,ostride,ocn,
						 op);
	}

	return op;
}

template<typename _SrcValT0,typename _SrcValT1,typename _SrcValT2,typename _DestValT,typename _MaskValT,typename _OpT,typename _MaskOpT>
inline _OpT& for_masked_pixel_3_1(const _SrcValT0 *pSrc0,const int width,const int height,const int istride0,const int icn0,
							 const _SrcValT1 *pSrc1,const int istride1,const int icn1,
							 const _SrcValT2 *pSrc2,const int istride2,const int icn2,
							_DestValT *pDest,const int ostride,const int ocn,
							_OpT &op, const _MaskValT *pMask,const int mstride,
							_MaskOpT &maskOp
							)
{
	if(pSrc0 && pSrc1 && pSrc2 && pDest && pMask)
	{
		for(int yi=0;yi<height;++yi,pSrc0+=istride0,pSrc1+=istride1,pSrc2+=istride2,pDest+=ostride,pMask+=mstride)
		{
			const _SrcValT0 *pix0=pSrc0;
			const _SrcValT1 *pix1=pSrc1;
			const _SrcValT2 *pix2=pSrc2;
			_DestValT *pdx=pDest;

			for(int xi=0;xi<width;++xi,pix0+=icn0,pix1+=icn1,pix2+=icn2,pdx+=ocn)
			{
				if(maskOp(pMask[xi]))
					op(pix0,pix1,pix2,pdx);
			}
		}
	}

	return op;
}

template<int cn, typename _PValT, typename _OpT>
class PointBinder_1
{
public:
	_PValT *m_img;
	int     m_width, m_height, m_stride;
	_OpT	m_op;
public:
	PointBinder_1(_PValT *img, int width, int height, int stride, _OpT op)
		:m_img(img),m_width(width),m_height(height),m_stride(stride),m_op(op)
	{
	}
	void operator()(int x, int y) 
	{
		m_op(m_img+y*m_stride+x*cn);
	}
};

template<int cn, typename _PValT, typename _OpT>
inline PointBinder_1<cn,_PValT,_OpT> PointBinder(_PValT *img, int width, int height, int stride, _OpT op)
{
	return PointBinder_1<cn,_PValT,_OpT>(img,width,height,stride,op);
}

template<bool check_boundary, typename _PointT, typename _OpT>
inline void for_line_pixels(int width, int height, _PointT start, _PointT end,  _OpT &op)
{
	double dx=end[0]-start[0], dy=end[1]-start[1];
	double L=sqrt(dx*dx+dy*dy);
	int np=(int)(L+0.5);

	if(np>0)
	{
		dx/=L;
		dy/=L;

		double x=start[0], y=start[1];
		if(false)//!check_boundary)
		{
			for(int i=0; i<np; ++i, x+=dx, y+=dy)
			{
				op( int(x+0.5), int(y+0.5) );
			}
		}
		else
		{
			for(int i=0; i<np; ++i, x+=dx, y+=dy)
			{
				int ix=int(x+0.5), iy=int(y+0.5);
				if( ( (uint)ix<(uint)width && (uint)iy<(uint)height ) )
				{
					op(ix,iy);
				}
			}
		}
	}
}

_IPF_END


#endif

