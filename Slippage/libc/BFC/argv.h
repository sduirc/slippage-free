

#ifndef _FF_INC_ARGV_H
#define _FF_INC_ARGV_H


#include"ffdef.h"
#include<string>
#include<vector>

_FF_BEG



enum
{
	ASF_NOT_FOUND=0x01
};

class _BFC_API IArgSet
{
	bool  m_bAllowExcept;
public:
	IArgSet();

	//query the value of the varible @var.
	//this is a pure virtual function that must be implemented by varible manager classes.
	virtual const char_t* const* Query(const char_t *var, int count=1, int pos=0,int *flag=NULL)=0;

	virtual bool Contain(const char_t *var);

	//call this function if any error occur.
	void OnFailed(const char_t *var);

	//call this function if an exception should be raised.
	void OnException(const char_t *var);

	//whether allow exception raised from this class. 
	//by default @OnFailed call @OnException to raise a standard exception derived from std::exception, its behavior can be changed
	//by calling this function with @bAllow be false.
	void AllowException(bool bAllow);

	//get the value of a varible with different types.
	//@var: the name of the varible.
	//@val: the array to store the varible.
	//@count: number of elements to get, with this argument array can be retrieved easily.
	//		e.g. to read -color 255 255 255, please call as: ->Get("color", &val[0], 3, 0);
	//@pos : the start position, with this argument the varibles can be defined to be structure.
	//		e.g. to specify a set of strings whose number is not fixed, you can use the following command:
	//			-str 5 str0 str1 str2 str3 str4
	//			 then use the following code to get:
	//		{
	//			int   count=pvar->Get<int>("str");
	//			string_t *pstr=new string_t[count];
	//			pvar->Get("str",pstr,count,2);
	//		}
	//@return value: the number of elements read successfully.
	//note that if exception is allowed and the required number of elements are not all read successfully, exceptions would be raised
	//before the function return, in which case you need not to check the return value to handle error.
	int Get(const char_t *var, bool *val, int count=1, int pos=0);

	int Get(const char_t *var, char_t *val, int count=1, int pos=0);

	int Get(const char_t *var, int *val, int count=1, int pos=0);

	int Get(const char_t *var, float *val, int count=1, int pos=0);

	int Get(const char_t *var, double *val, int count=1, int pos=0);

	int Get(const char_t *var, string_t *val, int count=1, int pos=0);

	int Get(const char_t *var, unsigned char *val, int count=1, int pos=0);

	//convenient function to read varibles with only one element.
	//this function always raise exception when read failed.
	template<typename _ValT>
	_ValT Get(const char_t *var, int pos=0)
	{
		_ValT val;
		if(this->Get(var,&val,1,pos)!=1)
			this->OnException(var);
		return val;
	}
	//convenient function to read varibles with default value.
	//this function would never raise exception, and the specified default value will be returned if failed to read from command line.
	template<typename _ValT>
	_ValT GetWithDef(const char_t *var,const _ValT &defaultVal, int pos=0)
	{
		bool bae=m_bAllowExcept;
		this->AllowException(false);

		_ValT val;
		int nr=this->Get(var,&val,1,pos);
		
		this->AllowException(bae);
		return nr==1? val:defaultVal;
	}
	//convenient function to read a c-style array.
	template<int _N, typename _ValT>
	int GetArray(const char_t *var, _ValT (&val)[_N], int pos=0)
	{
		return this->Get(var, val, _N, pos);
	}
	
#if 0
	//the vector can be represented as a structure, with the first element be the number of elements to read.
	//return value: negative if failed.
	template<typename _ValT,typename _AllocT>
	int GetVector(const char_t *var, std::vector<_ValT,_AllocT> &vec)
	{
		std::vector<_ValT,_AllocT> temp;
		int count=0;

		if(this->Get(var,&count,1,0)==1)
		{
			if(count>0)
			{
				temp.resize(count);

				if(this->Get(var,&temp[0],count,1)==count)
					vec.swap(temp);
				else
					count=-1;
			}
			else
				vec.resize(0);
		}
		else
			count=-1;

		if(count<0)
			this->OnFailed(var);

		return count;
	}
#else

	//need not to specify the number of elements
	template<typename _ValT,typename _AllocT>
	int GetVector(const char_t *var, std::vector<_ValT,_AllocT> &vec)
	{
		std::vector<_ValT,_AllocT> temp;

		bool bae=m_bAllowExcept;
		this->AllowException(false);

		_ValT vx;

		for(int i=0; ; ++i)
		{
			if(this->Get(var,&vx,1,i)==1)
			{
				temp.push_back(vx);
			}
			else
				break;
		}

		vec.swap(temp);
		
		this->AllowException(bae);

		return (int)vec.size();
	}

#endif
};

/*
to combine multiple arg-sets
*/
class _BFC_API IArgSetCombined
	:public IArgSet
{
protected:
	IArgSet  *m_pNext;
public:
	IArgSetCombined();

	virtual void SetNext(IArgSet *pNext);
};

//argument set used with command line.
class _BFC_API CommandArgSet
	:public IArgSetCombined
{
	class _CImp;

	_CImp  *m_pImp;

public:
	
//	VFX_DECLARE_COPY_FUNC(CommandArgSet);

	CommandArgSet();

	~CommandArgSet();

//	const vfxArgInf* Find(const char_t *var);

	//get the number of argument.
	int  NArg() const;

//	const vfxArgInf* GetArgInf(int idx);

	//set command line
	virtual void SetArg(const string_t &arg);

	void  SetArg(int argc, char_t *argv[]);

public:
	virtual const char_t* const* Query(const char_t *var, int count=1, int pos=0,int *flag=NULL);
};


_FF_END

#endif

