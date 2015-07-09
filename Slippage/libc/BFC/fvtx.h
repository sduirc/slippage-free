
#ifndef _INC_FVTX_H
#define _INC_FVTX_H

#include"ffdef.h"
#include"bfc/err.h"

#include<string>

_FF_BEG


/*
this class allow you to define an error table in c-style
*/
struct _BFC_API tagFVTError
{
public:
	std::string   m_err;
	std::string   m_msg;
};

struct _BFC_API FVTError
	:public tagFVTError
{
public:
	FVTError();

	FVTError(const std::string &err, const std::string &msg);

	std::string  GetErrorMessage(const std::string &arg) const;
};

class _BFC_API FVTErrorSet
{
	class _CImp;

	_CImp *m_pImp;
public:
	FVTErrorSet();

	~FVTErrorSet();

	FVTErrorSet(const FVTErrorSet &right);

	void   Swap(FVTErrorSet &right);

	FVTErrorSet& operator=(const FVTErrorSet &right);

	void RegisterError(const FVTError *errArray, int count);

	const FVTError* QueryError(const std::string &sig) const;
};

_BFC_API void  ParseErrorString(const std::string &contents, FVTErrorSet &errorSet);

_BFC_API void  ParseErrorFile(const std::string &file, FVTErrorSet &errorSet);

class _BFC_API fvt_auto_register_error
{
public:
	fvt_auto_register_error(const FVTError *errArray, int count);
};

#define FVT_REGISTER_DEFAULT_ERROR(table)  static ff::fvt_auto_register_error   _auto_register_error_##table((ff::FVTError*)&(table)[0], sizeof(table)/sizeof(FVTError));

_BFC_API FVTErrorSet* PushErrorSet();

_BFC_API void  PushErrorSet(const FVTErrorSet &errSet);

_BFC_API void  PopErrorSet();

_BFC_API const FVTError * QueryError(const std::string &err);

_BFC_API std::string  GetErrorMessage(const char *err, const std::string &arg);



_FF_END


#endif

