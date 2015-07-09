
#ifndef _FF_CSS_MEM_H
#define _FF_CSS_MEM_H

#include"ffdef.h"

struct _ff_debug_new;


_FFS_API void* __cdecl operator new (size_t size,const char *file,int line,_ff_debug_new *);

_FFS_API void* __cdecl operator new [] (size_t size,const char *file,int line,_ff_debug_new *);

_FFS_API void __cdecl operator delete(void* p,  const char *file, int line,_ff_debug_new *);

_FFS_API void __cdecl operator delete[](void* p, const char *file, int line,_ff_debug_new *);



#ifdef _DEBUG

#define FF_DEBUG_NEW new(__FILE__,__LINE__,(_ff_debug_new *)NULL)

#else

#define FF_DEBUG_NEW new

#endif


_FF_BEG

_FFS_API void* ff_alloc(size_t size);

_FFS_API void  ff_free(void *ptr);


_FFS_API void* ff_galloc(size_t size);

_FFS_API void  ff_gfree(void *ptr);

typedef void (*ff_gfree_func_t)();

_FFS_API void  ff_add_gfree_func(ff_gfree_func_t fp);

_FFS_API void  ff_gfree_all();

_FFS_API void enable_memory_leak_report(bool enable=true);


class _FFS_API ff_mem
{
	void *m_mem;
public:
	ff_mem();

	void set_mem(void *mem);

	~ff_mem();
};

_FFS_API char* ff_w2a(const wchar_t *wcs, ff_mem &mem);

_FFS_API wchar_t* ff_a2w(const char *acs, ff_mem &mem);




_FF_END

#endif


