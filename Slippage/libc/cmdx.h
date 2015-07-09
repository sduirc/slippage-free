
#ifndef _INC_CMDX_H
#define _INC_CMDX_H

#ifndef _CMDX_API

#ifdef CMDX_EXPORTS
#define _CMDX_API __declspec(dllexport)
#else
#define _CMDX_API __declspec(dllimport)
#endif

#endif

#ifdef __cplusplus
extern "C" {
#endif

enum
{
//	CMDX_EXEC_EXE = 1,
//	CMDX_EXEC_DLL = 2,

	CMDX_GET_VAR,
	CMDX_SET_VAR,

	CMDX_GET_VAR_INT,
	CMDX_SET_VAR_INT,

	CMDX_GET_ERRNO,
	CMDX_SET_ERRNO,

	CMDX_INIT_CONSOLE = 100,
	CMDX_PROCESS_PFREG = 101,
};

typedef _CMDX_API int ( *ft_cmdx_proc)(int,int,void*);

_CMDX_API int  cmdx_proc(int cmd, int param, void *data);


#ifdef __cplusplus
}
#endif










#endif



