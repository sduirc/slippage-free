
#ifndef _FF_BFC_FIO_H
#define _FF_BFC_FIO_H

#include"ffdef.h"
#include"bfc\bfstream.h"
#include<string>
#include<vector>

_FF_BEG

/*磁盘交换文件
*/
class _BFC_API DiskSwapFile
{
	class _CImp;

	_CImp *m_imp;
public:

	DiskSwapFile();

	~DiskSwapFile();

	void Open(const string_t &file, bool bnew=true);

	void Save();

	void Close();

	void SaveBlock(const void *buf, int size, int *id);

	void SetBlockName(int id, const char_t *name,bool allowMultiple=true);

	int  IDFromName(const char_t *name);

	int  QueryBlockSize(int id);

	int  LoadBlock(void *buf, int size, int id);

	bool DeleteBlock(int id);
};

class _BFC_API  NamedStreamFile
{
	class _CImp;

	_CImp  *m_imp;
public:
	
	NamedStreamFile(const char magic[]=NULL);

	~NamedStreamFile();

	void  Load(const string_t &file, bool bnew=true);

	const void* GetUserHead(int &size);

	//return size copied into @buf
	int   GetUserHead(void *buf, int buf_size);

	void  SetUserHead(const void *head, int size);

	BMStream *GetStream(const std::wstring &name, bool bnew=true);

	void  DeleteStream(const std::wstring &name);

	enum
	{
		SAVE_TRANSFORM=0x01,
	};
	void  Save(const char_t *file=NULL, int flag=0);

	void  ListStreams(std::vector<std::wstring> &vname);

	/*保存数据前对数据进行变换（压缩）

	@dsize : 输入@dest的大小，输出变换后的数据大小
	
	@dir   : >0 正向（压缩）；<0 逆向（解压）

	@RV    : >0 成功； =0 无动作，@dest未被修改；　<0 错误
	*/
	virtual int TransformData(const void *src, size_t isize, void *dest, size_t &dsize, int dir);
};

class _BFC_API PackedFile
	:public NamedStreamFile
{
public:
	PackedFile(const char magic[]=NULL);

//	int SetHead(const void* head, int size);

//	int GetHead(void *head, int size);

	//@vnames : if is not null, used as the name of each file stream; othersie @vfiles is used
	int AddFiles(const std::vector<string_t> & vfiles, const string_t &dir, const std::vector<string_t> *vnames=NULL);

	int Unpack(const string_t &dir, const string_t *backup_dir = NULL);
};

class _BFC_API  TextBlock
{
public:
	string_t		m_title;

	string_t     m_text;
};

//return the number of blocks
//return -1 if failed
/*
${title
text
//comment
$}
*/
int _BFC_API LoadTextBlocks(const string_t &file, std::vector<TextBlock> &vblk);

//return -1 if failed
/*
abc...\ (connected)
cdef 
//commented
*/
int _BFC_API LoadTextLines(const string_t &file, std::vector<string_t> &vlines);


int _BFC_API SplitText(const string_t &text, std::vector<string_t> &vsplit, char_t split_char);


/*HTB=Hierachical Text Blocks

<tag1>
 <tag11> the value of tag11 </tag11>
 <tag12>
 //comments :...
 // % : special char.
  <tag121> the value %<> of tag121 </tag121>
 </tag12>
</tag1>

<tag2>
 the value of tag2 
</tag2>

*/
class _BFC_API HTBFile
{
	class _CImp;

	_CImp  *m_imp;
public:
	HTBFile();

	~HTBFile();

	int Load(const string_t &file);

	const string_t *GetBlock(const string_t &path) const;

	const string_t& GetFile() const;
};



_FF_END


#endif

