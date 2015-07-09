
#ifndef _FF_BFC_ARRAY_H
#define _FF_BFC_ARRAY_H

#include"ffdef.h"
#include"bfc\ctc.h"

//#define CTCAssert(...) 

_FF_BEG

template<typename _ValT,int size>
class Array
{
protected:
	typedef typename SelType<const _ValT&,_ValT,IsClass<_ValT>::Yes>::RType _CParamT; //const parameter type.
public:
	typedef _ValT ValueType;

	enum{SIZE=size};

public:
	//default constructor do nothing.
	//note if @_ValT is a compiler inner type such as int,float,etc., array is not initialized to zero.
	Array()
	{
	}
	//initialize all elements to @vx.
	explicit Array(_CParamT vx)
	{
		for(int i=0;i<size;++i)
			m_Arr[i]=vx;
	}
	//the following constructors are used to initialize the array with specific size.
	//note if you call the constructor with more or less arguments than the array required, the
	//@CTCAssert assertion in the constructor would failed and a compiler error would present,
	//for example:
	//	Array<int,3> arr(1,2);
	//is not valid, this is useful to prevent you from misusing partial initialized array.

	Array(_CParamT v0,_CParamT v1)
	{
		CTCAssert(size==2);
		m_Arr[0]=v0,m_Arr[1]=v1;
	}
	Array(_CParamT v0,_CParamT v1,_CParamT v2)
	{
		CTCAssert(size==3);
		m_Arr[0]=v0;m_Arr[1]=v1;m_Arr[2]=v2;
	}
	Array(_CParamT v0,_CParamT v1,_CParamT v2,_CParamT v3)
	{
		CTCAssert(size==4);
		m_Arr[0]=v0;m_Arr[1]=v1;m_Arr[2]=v2;m_Arr[3]=v3;
	}
	Array(_CParamT v0,_CParamT v1,_CParamT v2,_CParamT v3,_CParamT v4)
	{
		CTCAssert(size==5);
		m_Arr[0]=v0;m_Arr[1]=v1;m_Arr[2]=v2;m_Arr[3]=v3;m_Arr[4]=v4;
	}
	Array(_CParamT v0,_CParamT v1,_CParamT v2,_CParamT v3,_CParamT v4,_CParamT v5)
	{
		CTCAssert(size==6);
		m_Arr[0]=v0;m_Arr[1]=v1;m_Arr[2]=v2;m_Arr[3]=v3;m_Arr[4]=v4;m_Arr[5]=v5;
	}

	//convert from array of other type with the same size.
	template<typename _Tx>
	explicit Array(const Array<_Tx,size>& right)
	{
		for(int i=0;i<size;++i)
			m_Arr[i]=static_cast<_ValT>(right[i]);
	}
	//element access.
	const _ValT& operator[](int idx) const
	{
		return m_Arr[idx];
	}
	_ValT& operator[](int idx)
	{
		return m_Arr[idx];
	}
protected:
	_ValT	m_Arr[size];
};


//equal test of array of the same type.
template<typename _ValT,int size>
inline bool 
operator==(const Array<_ValT,size>& left,const Array<_ValT,size>& right)
{
	for(int i=0;i<size;++i)
		if(left[i]!=right[i])
			return false;
	return true;
}
template<typename _ValT,int size>
inline bool 
operator!=(const Array<_ValT,size>& left,const Array<_ValT,size>& right)
{
	return !(left==right);
}

template<typename _ValT>
inline bool 
ArrEqual(const _ValT *left,const _ValT *right,const int dim)
{
	for(int i=0;i<dim;++i)
		if(left[i]!=right[i])
			return false;
	return true;
}

template<typename _ValT>
inline void
ArrDimMax(const _ValT *pArr,const int dim,const int count,_ValT *pMax)
{
	if(pArr)
	{
		memcpy(pMax,pArr,sizeof(_ValT)*dim);
		pArr+=dim;
		for(int i=1;i<count;pArr+=dim,++i)
		{
			for(int j=0;j<dim;++j)
			{
				if(pArr[j]>pMax[j])
					pMax[j]=pArr[j];
			}
		}
	}
}
template<typename _ArrT>
inline void
ArrDimMax(const _ArrT *pArr,const int count,_ArrT &dimMax)
{
	if(pArr)
		ArrDimMax(&pArr[0][0],_ArrT::SIZE,count,&dimMax[0]);
}
template<typename _ValT>
inline void
ArrDimMin(const _ValT *pArr,const int dim,const int count,_ValT *pMin)
{
	if(pArr)
	{
		memcpy(pMin,pArr,sizeof(_ValT)*dim);
		pArr+=dim;
		for(int i=1;i<count;pArr+=dim,++i)
		{
			for(int j=0;j<dim;++j)
			{
				if(pArr[j]<pMin[j])
					pMin[j]=pArr[j];
			}
		}
	}
}
template<typename _ArrT>
inline void
ArrDimMin(const _ArrT *pArr,const int count,_ArrT &dimMin)
{
	if(pArr)
		ArrDimMin(&pArr[0][0],_ArrT::SIZE,count,&dimMin[0]);
}

_FF_END

#endif

