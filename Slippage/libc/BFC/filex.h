
#ifndef _FVT_INC_FILEEX_H
#define _FVT_INC_FILEEX_H

#include"ffdef.h"

#include<string>
#include<vector>

#include<windows.h>

_FF_BEG

class _BFC_API IFileAPI
{
public:
	bool IsDirectory(LPCTSTR lpFileName);
	
	void RemoveDirectoryRecursively(LPCTSTR lpPath);
	
	void CopyDirectory(LPCTSTR lpSrcDir,LPCTSTR lpDestDir,bool bFailIfExist=false);

	void Copy(LPCTSTR lpSrc,LPCTSTR lpDest,bool bFailIfExist=false);

	void Remove(LPCTSTR lpPath);

public:
	virtual bool   IsExist(LPCTSTR lpFileName) =0;

	virtual DWORD  GetFileAttributes(LPCTSTR lpFileName) =0;

	virtual HANDLE FindFirstFile(LPCTSTR lpFileName,LPWIN32_FIND_DATA lpFindFileData) =0;

	virtual bool FindNextFile( HANDLE hFindFile, LPWIN32_FIND_DATA lpFindFileData ) =0;

	virtual void FindClose(HANDLE hFindFile) =0;

	virtual void CopyFile(LPCTSTR lpSrcFile,LPCTSTR lpDestFile,BOOL bFailIfExists) =0;

	virtual void DeleteFile(LPCTSTR lpFileName) =0;

	virtual void RemoveDirectory(LPCTSTR lpPathName) =0;

	virtual void CreateDirectory(LPCTSTR lpPathName,LPSECURITY_ATTRIBUTES lpSecurityAttributes=NULL) =0;

	virtual ~IFileAPI() throw();
};


class _BFC_API LocalFileAPI
	:public IFileAPI
{
public:
	static LocalFileAPI global;
public:
	virtual bool   IsExist(LPCTSTR lpFileName);

	virtual DWORD  GetFileAttributes(LPCTSTR lpFileName);

	virtual HANDLE FindFirstFile(LPCTSTR lpFileName,LPWIN32_FIND_DATA lpFindFileData);

	virtual bool FindNextFile( HANDLE hFindFile, LPWIN32_FIND_DATA lpFindFileData );

	virtual void FindClose(HANDLE hFindFile);

	virtual void CopyFile(LPCTSTR lpSrcFile,LPCTSTR lpDestFile,BOOL bFailIfExists);

	virtual void DeleteFile(LPCTSTR lpFileName);

	virtual void RemoveDirectory(LPCTSTR lpPathName);

	virtual void CreateDirectory(LPCTSTR lpPathName,LPSECURITY_ATTRIBUTES lpSecurityAttributes=NULL);
};



class _BFC_API IForEachFileCallBack
{
protected:
	string_t m_strRoot;
public:
	//set root directory.
	virtual void OnSetRoot(const string_t &root);
	//enter directory.
	//@relativePath : path relative to root directory @m_strRoot.
	virtual bool OnEnterDir(const string_t &relativePath);

	virtual bool OnDirFound(const string_t &relativePath);
	
	virtual bool OnFileFound(const string_t &relativePath,const string_t &fileName,const WIN32_FIND_DATA *pFFD);

	virtual void OnLeaveDir(const string_t &relativePath);
};


void _BFC_API ForEachFile(const string_t &dir,IForEachFileCallBack *pOp,bool bRecursive=true,IFileAPI *pFileAPI=NULL);


//=========================================================================================================

class _BFC_API FileFindData
	:public WIN32_FIND_DATA
{
public:
	FileFindData(const WIN32_FIND_DATA *pData=NULL);

	bool IsDir() const
	{
		return (dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)!=0;
	}

//public:
//	friend void BFSRead(IBStream &is,FileFindData &val);
//
//	friend void BFSWrite(OBStream &os,const FileFindData &val);
};


void _BFC_API ListFiles(const string_t &dir,std::vector<FileFindData> &vList,DWORD excAtt=0,IFileAPI *pFileAPI=NULL);
										
void _BFC_API ListSubFolders(const string_t &dir,std::vector<string_t> &vList,IFileAPI *pFileAPI=NULL);



_FF_END



#endif


