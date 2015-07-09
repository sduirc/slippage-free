
#ifndef _INC_ADIS_H
#define _INC_ADIS_H

#include"iff/image.h"
#include"BFC\autores.h"

inline int make_ufid(int videoID, int frameID)
{
	assert(uint(videoID)<=255);
	return (videoID<<24)|frameID;
}


class ImageSetReader
{
	int m_id;
	cv::Size  m_dsize;
public:

	void  SetID(int id)
	{
		m_id=id;
	}

	int   GetID() const
	{
		return m_id;
	}

	Mat* Read(int pos)
	{
		this->SetPos(pos);
		return this->Read();
	}

	bool ReadSized(int pos, cv::Mat &img, const cv::Size &dsize);

	void SetDSize(const cv::Size &dsize)
	{
		m_dsize=dsize;
	}

	bool ReadSized(int pos, cv::Mat &img);

	virtual int  GetUFID(int pos) =0;

	//width of current image.
	virtual int		Width() =0;
	//height of current image.
	virtual int		Height() =0;
	//read current image without move to the next.
	//return NULL if end of image set reached.
	virtual Mat* Read()  =0;

	//move read pos forward
	virtual bool     MoveForward() =0;

	//number of images in the image set.
	virtual int		Size() =0;
	//get current pos.
	virtual int		 Pos() =0;
	//set current pos.
	virtual int	 SetPos(int pos) =0;

	virtual string_t FrameName(int pos) =0;

	ImageSetReader();

	virtual ~ImageSetReader();

private:
	ImageSetReader(const ImageSetReader&);
	ImageSetReader& operator=(const ImageSetReader&);
};

#if 0
struct capxCapture;

class ISRVideo
	:public ImageSetReader
{
protected:
	capxCapture			*m_pCap;

	int					m_width,m_height;
	Mat  			   *m_pCurImg;
public:
	ISRVideo();
	
	int  Create(const string_t &file);

	~ISRVideo();

public:
	virtual int     GetUFID(int pos);

	virtual int		Width();
	
	virtual int		Height();
	
	virtual Mat* Read();
	
	virtual bool     MoveForward();

	virtual int		Size();

	virtual int		 Pos();
	
	virtual int	 SetPos(int pos);

	virtual string_t FrameName(int pos);
};
#endif

class ISRImages
	:public ImageSetReader
{
public:
	typedef std::pair<string_t,uint>  FilePairT;
protected:
	int		m_width, m_height;

	int			m_bufPos;
	Mat			m_bufImg;

	int			m_pos; 
	string_t		m_dir;
	std::vector<FilePairT>  m_vfiles;
	int			m_nc; //force number of channels
public:
	ISRImages();

	//@nc : convert number of channels if @nc>0
	int		Create(const string_t &file, int nc);

	int		GetFileIndex(const string_t &name);

public:
	virtual int     GetUFID(int pos);

	virtual int		Width();
	
	virtual int		Height();
	
	virtual Mat* Read();
	
	virtual bool     MoveForward();

	virtual int		Size();

	virtual int		 Pos();
	
	virtual int	 SetPos(int pos);

	virtual string_t FrameName(int pos);

};



typedef ff::AutoPtr<ImageSetReader>  _ISRPtrT;

#endif

