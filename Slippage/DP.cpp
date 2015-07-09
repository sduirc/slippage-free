
#include"stdafx.h"

#include"DP.h"
#include"dataset.h"
#include"core.h"

#include"functions.h"

#include"SutherlandHodgman.h"



namespace{
//=====================================================
#if 0
#define M 9 // rows
#define N 9 // cols

double SIGN(double a, double b)
{
    if(b > 0) {
        return fabs(a);
    }

    return -fabs(a);
}

double PYTHAG(double a, double b)
{
    double at = fabs(a), bt = fabs(b), ct, result;

    if (at > bt)       { ct = bt / at; result = at * sqrt(1.0 + ct * ct); }
    else if (bt > 0.0) { ct = at / bt; result = bt * sqrt(1.0 + ct * ct); }
    else result = 0.0;
    return(result);
}

// Returns 1 on success, fail otherwise
int dsvd(double *a, int m, int n, double *w, double *v)
{
    int flag, i, its, j, jj, k, l, nm;
    double c, f, h, s, x, y, z;
    double anorm = 0.0, g = 0.0, scale = 0.0;
    double rv1[N];

    if (m < n)
    {
        //fprintf(stderr, "#rows must be > #cols \n");
        return(-1);
    }

    //rv1 = (double *)malloc((unsigned int) n*sizeof(double));

/* Householder reduction to bidiagonal form */
    for (i = 0; i < n; i++)
    {
        /* left-hand reduction */
        l = i + 1;
        rv1[i] = scale * g;
        g = s = scale = 0.0;
        if (i < m)
        {
            for (k = i; k < m; k++)
                scale += fabs((double)a[k*n+i]);
            if (scale)
            {
                for (k = i; k < m; k++)
                {
                    a[k*n+i] = (double)((double)a[k*n+i]/scale);
                    s += ((double)a[k*n+i] * (double)a[k*n+i]);
                }
                f = (double)a[i*n+i];
                g = -SIGN(sqrt(s), f);
                h = f * g - s;
                a[i*n+i] = (double)(f - g);
                if (i != n - 1)
                {
                    for (j = l; j < n; j++)
                    {
                        for (s = 0.0, k = i; k < m; k++)
                            s += ((double)a[k*n+i] * (double)a[k*n+j]);
                        f = s / h;
                        for (k = i; k < m; k++)
                            a[k*n+j] += (double)(f * (double)a[k*n+i]);
                    }
                }
                for (k = i; k < m; k++)
                    a[k*n+i] = (double)((double)a[k*n+i]*scale);
            }
        }
        w[i] = (double)(scale * g);

        /* right-hand reduction */
        g = s = scale = 0.0;
        if (i < m && i != n - 1)
        {
            for (k = l; k < n; k++)
                scale += fabs((double)a[i*n+k]);
            if (scale)
            {
                for (k = l; k < n; k++)
                {
                    a[i*n+k] = (double)((double)a[i*n+k]/scale);
                    s += ((double)a[i*n+k] * (double)a[i*n+k]);
                }
                f = (double)a[i*n+l];
                g = -SIGN(sqrt(s), f);
                h = f * g - s;
                a[i*n+l] = (double)(f - g);
                for (k = l; k < n; k++)
                    rv1[k] = (double)a[i*n+k] / h;
                if (i != m - 1)
                {
                    for (j = l; j < m; j++)
                    {
                        for (s = 0.0, k = l; k < n; k++)
                            s += ((double)a[j*n+k] * (double)a[i*n+k]);
                        for (k = l; k < n; k++)
                            a[j*n+k] += (double)(s * rv1[k]);
                    }
                }
                for (k = l; k < n; k++)
                    a[i*n+k] = (double)((double)a[i*n+k]*scale);
            }
        }
        anorm = max(anorm, (fabs((double)w[i]) + fabs(rv1[i])));
    }

    /* accumulate the right-hand transformation */
    for (i = n - 1; i >= 0; i--)
    {
        if (i < n - 1)
        {
            if (g)
            {
                for (j = l; j < n; j++)
                    v[j*n+i] = (double)(((double)a[i*n+j] / (double)a[i*n+l]) / g);
                    /* double division to avoid underflow */
                for (j = l; j < n; j++)
                {
                    for (s = 0.0, k = l; k < n; k++)
                        s += ((double)a[i*n+k] * (double)v[k*n+j]);
                    for (k = l; k < n; k++)
                        v[k*n+j] += (double)(s * (double)v[k*n+i]);
                }
            }
            for (j = l; j < n; j++)
                v[i*n+j] = v[j*n+i] = 0.0;
        }
        v[i*n+i] = 1.0;
        g = rv1[i];
        l = i;
    }

    /* accumulate the left-hand transformation */
    for (i = n - 1; i >= 0; i--)
    {
        l = i + 1;
        g = (double)w[i];
        if (i < n - 1)
            for (j = l; j < n; j++)
                a[i*n+j] = 0.0;
        if (g)
        {
            g = 1.0 / g;
            if (i != n - 1)
            {
                for (j = l; j < n; j++)
                {
                    for (s = 0.0, k = l; k < m; k++)
                        s += ((double)a[k*n+i] * (double)a[k*n+j]);
                    f = (s / (double)a[i*n+i]) * g;
                    for (k = i; k < m; k++)
                        a[k*n+j] += (double)(f * (double)a[k*n+i]);
                }
            }
            for (j = i; j < m; j++)
                a[j*n+i] = (double)((double)a[j*n+i]*g);
        }
        else
        {
            for (j = i; j < m; j++)
                a[j*n+i] = 0.0;
        }
        ++a[i*n+i];
    }

    /* diagonalize the bidiagonal form */
    for (k = n - 1; k >= 0; k--)
    {                             /* loop over singular values */
        for (its = 0; its < 30; its++)
        {                         /* loop over allowed iterations */
            flag = 1;
            for (l = k; l >= 0; l--)
            {                     /* test for splitting */
                nm = l - 1;
                if (fabs(rv1[l]) + anorm == anorm)
                {
                    flag = 0;
                    break;
                }
                if (fabs((double)w[nm]) + anorm == anorm)
                    break;
            }
            if (flag)
            {
                c = 0.0;
                s = 1.0;
                for (i = l; i <= k; i++)
                {
                    f = s * rv1[i];
                    if (fabs(f) + anorm != anorm)
                    {
                        g = (double)w[i];
                        h = PYTHAG(f, g);
                        w[i] = (double)h;
                        h = 1.0 / h;
                        c = g * h;
                        s = (- f * h);
                        for (j = 0; j < m; j++)
                        {
                            y = (double)a[j*n+nm];
                            z = (double)a[j*n+i];
                            a[j*n+nm] = (double)(y * c + z * s);
                            a[j*n+i] = (double)(z * c - y * s);
                        }
                    }
                }
            }
            z = (double)w[k];
            if (l == k)
            {                  /* convergence */
                if (z < 0.0)
                {              /* make singular value nonnegative */
                    w[k] = (double)(-z);
                    for (j = 0; j < n; j++)
                        v[j*n+k] = (-v[j*n+k]);
                }
                break;
            }
            if (its >= 30) {
                //free((void*) rv1);
                //fprintf(stderr, "No convergence after 30,000! iterations \n");
                return(0);
            }

            /* shift from bottom 2 x 2 minor */
            x = (double)w[l];
            nm = k - 1;
            y = (double)w[nm];
            g = rv1[nm];
            h = rv1[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = PYTHAG(f, 1.0);
            f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

            /* next QR transformation */
            c = s = 1.0;
            for (j = l; j <= nm; j++)
            {
                i = j + 1;
                g = rv1[i];
                y = (double)w[i];
                h = s * g;
                g = c * g;
                z = PYTHAG(f, h);
                rv1[j] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y = y * c;
                for (jj = 0; jj < n; jj++)
                {
                    x = (double)v[jj*n+j];
                    z = (double)v[jj*n+i];
                    v[jj*n+j] = (double)(x * c + z * s);
                    v[jj*n+i] = (double)(z * c - x * s);
                }
                z = PYTHAG(f, h);
                w[j] = (double)z;
                if (z)
                {
                    z = 1.0 / z;
                    c = f * z;
                    s = h * z;
                }
                f = (c * g) + (s * y);
                x = (c * y) - (s * g);
                for (jj = 0; jj < m; jj++)
                {
                    y = (double)a[jj*n+j];
                    z = (double)a[jj*n+i];
                    a[jj*n+j] = (double)(y * c + z * s);
                    a[jj*n+i] = (double)(z * c - y * s);
                }
            }
            rv1[l] = 0.0;
            rv1[k] = f;
            w[k] = (double)x;
        }
    }
    //free((void*) rv1);
    return(1);
}
#endif

void _set_identity(double d[], int m)
{
	for(int i=0; i<m; ++i)
	{
		for(int j=0; j<m; ++j)
		{
			d[m*i+j]=i==j? 1.0 : 0;
		}
	}
}

bool _inv_mat44(const double m[16], double d[16])
{
	const double a0=m[0], a1=m[1], a2=m[2], a3=m[3], a4=m[4], a5=m[5], a6=m[6], a7=m[7], a8=m[8], a9=m[9], a10=m[10], a11=m[11], a12=m[12], a13=m[13], a14=m[14], a15=m[15];

	double det=a0*a5*a10*a15 - a0*a5*a11*a14 - a0*a6*a9*a15 + a0*a6*a11*a13 + a0*a7*a9*a14 - a0*a7*a10*a13 - a1*a4*a10*a15 + a1*a4*a11*a14 + a1*a6*a8*a15 - a1*a6*a11*a12 - a1*a7*a8*a14 + a1*a7*a10*a12 + a2*a4*a9*a15 - a2*a4*a11*a13 - a2*a5*a8*a15 + a2*a5*a11*a12 + a2*a7*a8*a13 - a2*a7*a9*a12 - a3*a4*a9*a14 + a3*a4*a10*a13 + a3*a5*a8*a14 - a3*a5*a10*a12 - a3*a6*a8*a13 + a3*a6*a9*a12;

	if(fabs(det)<1e-8)
	{
//		_set_identity(d,4);
		return false;
	}

	double inv[16]=
	{
	a5*a10*a15 - a5*a11*a14 - a6*a9*a15 + a6*a11*a13 + a7*a9*a14 - a7*a10*a13, a1*a11*a14 - a1*a10*a15 + a2*a9*a15 - a2*a11*a13 - a3*a9*a14 + a3*a10*a13, a1*a6*a15 - a1*a7*a14 - a2*a5*a15 + a2*a7*a13 + a3*a5*a14 - a3*a6*a13, a1*a7*a10 - a1*a6*a11 + a2*a5*a11 - a2*a7*a9 - a3*a5*a10 + a3*a6*a9,
	a4*a11*a14 - a4*a10*a15 + a6*a8*a15 - a6*a11*a12 - a7*a8*a14 + a7*a10*a12, a0*a10*a15 - a0*a11*a14 - a2*a8*a15 + a2*a11*a12 + a3*a8*a14 - a3*a10*a12, a0*a7*a14 - a0*a6*a15 + a2*a4*a15 - a2*a7*a12 - a3*a4*a14 + a3*a6*a12, a0*a6*a11 - a0*a7*a10 - a2*a4*a11 + a2*a7*a8 + a3*a4*a10 - a3*a6*a8,
	a4*a9*a15 - a4*a11*a13 - a5*a8*a15 + a5*a11*a12 + a7*a8*a13 - a7*a9*a12,   a0*a11*a13 - a0*a9*a15 + a1*a8*a15 - a1*a11*a12 - a3*a8*a13 + a3*a9*a12, a0*a5*a15 - a0*a7*a13 - a1*a4*a15 + a1*a7*a12 + a3*a4*a13 - a3*a5*a12,   a0*a7*a9 - a0*a5*a11 + a1*a4*a11 - a1*a7*a8 - a3*a4*a9 + a3*a5*a8,
	a4*a10*a13 - a4*a9*a14 + a5*a8*a14 - a5*a10*a12 - a6*a8*a13 + a6*a9*a12,   a0*a9*a14 - a0*a10*a13 - a1*a8*a14 + a1*a10*a12 + a2*a8*a13 - a2*a9*a12, a0*a6*a13 - a0*a5*a14 + a1*a4*a14 - a1*a6*a12 - a2*a4*a13 + a2*a5*a12,   a0*a5*a10 - a0*a6*a9 - a1*a4*a10 + a1*a6*a8 + a2*a4*a9 - a2*a5*a8
	};

	for(int i=0; i<16; ++i)
		d[i]=inv[i]/det;

	return true;
}

bool _inv_mat33(const double m[9], double inv[9])
{
	const double a=m[0],b=m[1],c=m[2],d=m[3],e=m[4],f=m[5],g=m[6],h=m[7],i=m[8];

	double D=(-a*e*i+a*f*h+d*b*i-d*c*h-g*b*f+g*c*e);

	if(fabs(D)<1e-8)
	{
//		_set_identity(inv,3);
		return false;
	}

	D=1.0/D;
	inv[0]=-(e*i-f*h)*D, inv[1]=(b*i-c*h)*D, inv[2]=(-b*f+c*e)*D;
	inv[3]=(d*i-f*g)*D,  inv[4]=-(a*i-c*g)*D,inv[5]=-(-a*f+c*d)*D;
	inv[6]=-(d*h-e*g)*D, inv[7]=(a*h-b*g)*D, inv[8]= (-a*e+b*d)*D; 

	return true;
}

bool _inv_mat33( const cv::Matx33d &m, cv::Matx33d &inv)
{
	return _inv_mat33(m.val, inv.val);
}


template<int np>
bool get_tsr_matrix(const Point2f points1[], const Point2f points2[], const float wei[], cv::Matx33d &T)
{
	// 转换成解方程 A * X = B
	// X1 * S * cos - Y1 * sin + dx = X2;
	// X1 * sin + Y1 * S * cos + dy = Y2;

	enum{rows = 2 * np};
	cv::Matx<double, rows, 4> A;
	cv::Matx<double, rows, 1> B;

	for( unsigned int i=0; i<np; i++)
	{
		A(2*i, 0) = points1[i].x * wei[i];
		A(2*i, 1) = -points1[i].y * wei[i];
		A(2*i, 2) = wei[i];
		A(2*i, 3) = 0;

		A(2*i+1, 0) = points1[i].y * wei[i];
		A(2*i+1, 1) = points1[i].x * wei[i];
		A(2*i+1, 2) = 0;
		A(2*i+1, 3) = wei[i];

		B(2*i, 0) = points2[i].x * wei[i];
		B(2*i+1, 0) = points2[i].y * wei[i];
	}

//	solve( A, B, X, DECOMP_SVD ); 
	cv::Matx<double, 4, rows> AT(A.t());
	cv::Matx<double,4,1> BX(AT*B);

	cv::Matx<double,4,4> AAT(AT*A);
	if(!_inv_mat44(AAT.val, AAT.val))
		return false;

	BX=AAT*BX;

	double s_cos = BX(0);
	double s_sin = BX(1);
	double dx = BX(2);
	double dy = BX(3);
	//Mat tsr = (Mat_<double>(2, 3) << s_cos, -sin, dx, sin, s_cos, dy);
	T=cv::Matx33d(s_cos,-s_sin,dx,s_sin,s_cos,dy,0,0,1);

	return true;
}

template<int np>
bool get_ts_matrix(const Point2f points1[], const Point2f points2[], const float wei[], cv::Matx33d &T)
{
	// 转换成解方程 A * X = B
	// X1 * S * cos - Y1 * sin + dx = X2;
	// X1 * sin + Y1 * S * cos + dy = Y2;

	enum{rows = 2 * np};
	cv::Matx<double, rows, 3> A;
	cv::Matx<double, rows, 1> B;


	for( unsigned int i=0; i<np; i++)
	{
		A(2*i, 0) = points1[i].x * wei[i];
		A(2*i, 1) = wei[i];
		A(2*i, 2) = 0;

		A(2*i+1, 0) = points1[i].y * wei[i];
		A(2*i+1, 1) = 0;
		A(2*i+1, 2) = wei[i];

		B(2*i, 0) = points2[i].x * wei[i];
		B(2*i+1, 0) = points2[i].y * wei[i];
	}

	cv::Matx<double, 3, rows> AT(A.t());
	cv::Matx<double,3,1> BX(AT*B);

	cv::Matx<double,3,3> AAT(AT*A);
	if(!_inv_mat33(AAT.val, AAT.val))
		return false;

	BX=AAT*BX;

	double scale = BX(0);
	double dx = BX(1);
	double dy = BX(2);
	T=cv::Matx33d(scale,0,dx,0,scale,dy,0,0,1);

	return true;
}


template<int np>
bool get_tsr_matrix(const Point2f points1[], const Point2f points2[], cv::Matx33d &T)
{
	// 转换成解方程 A * X = B
	// X1 * S * cos - Y1 * sin + dx = X2;
	// X1 * sin + Y1 * S * cos + dy = Y2;

	enum{rows = 2 * np};
	cv::Matx<double, rows, 4> A;
	cv::Matx<double, rows, 1> B;

	for( unsigned int i=0; i<np; i++)
	{
		A(2*i, 0) = points1[i].x;
		A(2*i, 1) = -points1[i].y;
		A(2*i, 2) = 1;
		A(2*i, 3) = 0;

		A(2*i+1, 0) = points1[i].y;
		A(2*i+1, 1) = points1[i].x;
		A(2*i+1, 2) = 0;
		A(2*i+1, 3) = 1;

		B(2*i, 0) = points2[i].x;
		B(2*i+1, 0) = points2[i].y;
	}

//	solve( A, B, X, DECOMP_SVD ); 
	cv::Matx<double, 4, rows> AT(A.t());
	cv::Matx<double,4,1> BX(AT*B);

	cv::Matx<double,4,4> AAT(AT*A);
	if(!_inv_mat44(AAT.val, AAT.val))
		return false;

	BX=AAT*BX;

	double s_cos = BX(0);
	double s_sin = BX(1);
	double dx = BX(2);
	double dy = BX(3);
	//Mat tsr = (Mat_<double>(2, 3) << s_cos, -sin, dx, sin, s_cos, dy);
	T=cv::Matx33d(s_cos,-s_sin,dx,s_sin,s_cos,dy,0,0,1);

	return true;
}

template<int np>
bool get_ts_matrix(const Point2f points1[], const Point2f points2[], cv::Matx33d &T)
{
	// 转换成解方程 A * X = B
	// X1 * S * cos - Y1 * sin + dx = X2;
	// X1 * sin + Y1 * S * cos + dy = Y2;

	enum{rows = 2 * np};
	cv::Matx<double, rows, 3> A;
	cv::Matx<double, rows, 1> B;


	for( unsigned int i=0; i<np; i++)
	{
		A(2*i, 0) = points1[i].x;
		A(2*i, 1) = 1;
		A(2*i, 2) = 0;

		A(2*i+1, 0) = points1[i].y;
		A(2*i+1, 1) = 0;
		A(2*i+1, 2) = 1;

		B(2*i, 0) = points2[i].x;
		B(2*i+1, 0) = points2[i].y;
	}

	cv::Matx<double, 3, rows> AT(A.t());
	cv::Matx<double,3,1> BX(AT*B);

	cv::Matx<double,3,3> AAT(AT*A);
	if(!_inv_mat33(AAT.val, AAT.val))
		return false;

	BX=AAT*BX;

	double scale = BX(0);
	double dx = BX(1);
	double dy = BX(2);
	T=cv::Matx33d(scale,0,dx,0,scale,dy,0,0,1);

	return true;
}



//cv::Mat get_transform(const cv::Mat &T, const std::vector<cv::Point3f> &vpt, float *error=NULL, bool rotation=false)
//template<int np>
//bool get_translation2(const cv::Matx33d &_T, cv::Matx33d &_TX, cv::Point2f _vpt[], float *error=NULL, bool rotation=false)
//{
//	cv::Mat T(_T);
//	std::vector<cv::Point3f> vpt;
//	for(int i=0; i<np; ++i)
//		vpt.push_back(Point3f(_vpt[i].x, _vpt[i].y, 1.0f));
//
//	std::vector<cv::Point3f> ptx;
//	cv::transform(vpt,ptx,T);
//
//	cv::Mat TX;
//	std::vector<cv::Point2f> pt1,pt2;
//	for(size_t i=0; i<vpt.size(); ++i)
//	{
//		pt1.push_back(Point2f(vpt[i].x/vpt[i].z,vpt[i].y/vpt[i].z));
//		pt2.push_back(Point2f(ptx[i].x/ptx[i].z,ptx[i].y/ptx[i].z));
//	}
//
//	if(rotation)
//		TX=::get_tsr_matrix(pt1,pt2);
//	else
//		TX=::get_ts_matrix(pt1,pt2);
//
//	affine_to_homogeneous(TX);
//
//	if(error)
//	{
//		cv::transform(pt1,ptx,TX);
//		*error=0;
//		for(size_t i=0; i<pt1.size(); ++i)
//		{
//			cv::Point2f dv(ptx[i].x-pt2[i].x, ptx[i].y-pt2[i].y);
//			*error+=dv.dot(dv);
//		}
//		*error/=pt1.size();
//	}
//	_TX=TX;
//	return true;
//}

template<int np>
bool get_transform(const cv::Matx33d &T, cv::Matx33d &TX, cv::Point2f vpt[], float *error=NULL, bool rotation=false)
{
	cv::Point2f ptx[np];
	do_transform(T,vpt,ptx,np);

	bool ok=false;
	if(rotation)
		ok=get_tsr_matrix<np>(vpt,ptx,TX);
	else
		ok=get_ts_matrix<np>(vpt,ptx,TX);

	if(ok && error)
	{
		//cv::transform(pt1,ptx,TX);
		cv::Point2f dptx[np];
		do_transform(TX,vpt,dptx,np);
		*error=0;
		for(size_t i=0; i<np; ++i)
		{
			cv::Point2f dv(dptx[i]-ptx[i]);
			*error+=dv.dot(dv);
		}
		*error/=np;
	}

	return ok;
}

template<int np, int nfoot>
bool get_transform(const cv::Matx33d &T, cv::Matx33d &TX, cv::Point2f vpt[], const float *wei, float *error=NULL, bool rotation=false)
{
	cv::Point2f ptx[np];
	do_transform(T,vpt,ptx,np);

	bool ok=false;
	if(rotation)
		ok=get_tsr_matrix<np>(vpt,ptx,wei,TX);
	else
		ok=get_ts_matrix<np>(vpt,ptx,wei,TX);

	if(ok && error)
	{
		//cv::transform(pt1,ptx,TX);
		cv::Point2f dptx[np];
		do_transform(TX,vpt,dptx,np);
		*error=0;
		for(size_t i=0; i<nfoot; ++i)
		{
			cv::Point2f dv(dptx[i]-ptx[i]);
			*error+=dv.dot(dv);
		}
		*error=0.1*(*error)/nfoot;
	}

	return ok;
}

template<int np, int nps>
bool get_restricted_motion(const cv::Matx33d &T, cv::Matx33d &TX, const cv::Point2f vpt[], const cv::Point2f sdpt[], const float wei[], float *error=NULL, float smooth_error_weight=0.1, bool rotation=false)
{
	cv::Point2f ptx[np];
	do_transform(T,vpt,ptx,np-nps);

	for(int i=0; i<nps; ++i)
		ptx[np-nps+i]=sdpt[i];

	bool ok=false;
	if(rotation)
		ok=get_tsr_matrix<np>(vpt,ptx,wei,TX);
	else
		ok=get_ts_matrix<np>(vpt,ptx,wei,TX);

	if(ok && error)
	{
		//cv::transform(pt1,ptx,TX);
		cv::Point2f dptx[np];
		do_transform(TX,vpt,dptx,np);
		*error=0;
		for(size_t i=0; i<np; ++i)
		{
			cv::Point2f dv(dptx[i]-ptx[i]);
			float err=dv.dot(dv);
			if(i>np-nps)
				err*=smooth_error_weight;

			*error+=err;
		}
		*error/=np;
	}

	return ok;
}

//========================================================================================

#define _SAVE_TK  1

struct DPNode
{
public:
	int		m_father;
	float	m_error;
	DPError m_errors;
	Point2f m_bshape[3];
#if _SAVE_TK
	cv::Matx33d  m_Tk;
#endif
};

struct _DPTNode
{
	cv::Matx33d  m_A;
	cv::Point2f  m_bgTCorners[4];
//	cv::Point2f  m_bshape[3];
};




double _get_smooth_error(const cv::Point2f pt0[4], const cv::Point2f pt1[4])
{
	double error=0;
	for(size_t i=0; i<4; ++i)
	{
		cv::Point2f dv(pt0[i]-pt1[i]);
		error+=dv.dot(dv);
	}
	return error;
}


double _get_matching_error(const cv::Point2f fpt[], const int np, const cv::Matx33d &fT, const cv::Matx33d &bT)
{
	double err=0;
	for(int i=0; i<np; ++i)
	{
		cv::Matx31d pti(fpt[i].x, fpt[i].y, 1.0f);
		cv::Matx31d ft(fT*pti), bt(bT*pti);

		double dx=ft(0)/ft(2)-bt(0)/bt(2), dy=ft(1)/ft(2)-bt(1)/bt(2);
		err+=dx*dx+dy*dy;
	}
	return err;
}




template<int N, typename _ValT>
inline void arr_assign(_ValT (&dest)[N], const _ValT (&src)[N])
{
	for(int i=0; i<N; ++i)
		dest[i]=src[i];
}

struct DPKernelData
{
	cv::Size	 m_fgSize;
	cv::Point2f  m_footCorners[N_FOOT_NBRS];
	cv::Point2f  m_bgCorners[4];
	cv::Point2f  m_bgTCorners[4];
	double       m_bgTArea;
	cv::Point2f  m_fgCorners[4];
	double       m_fgArea;
	cv::Matx33d  bgT;
	cv::Matx33d  bgTInv;
	int			 m_motion[5]; 
	SutherlandHodgman *m_fgClipper;
	cv::Point2f       *m_fgClippedPolyBuffer;
	cv::Point2f       *m_fgTransformedPolyBuffer;
	float			  *m_vWeight;
public:
	DPKernelData()
		:m_vWeight(NULL)
	{
	}
};

void poly_clip(SutherlandHodgman *clipper, const cv::Point2f *vpt, int ninput, cv::Point2f *cpt, int &nclipped)
{
	clipper->Clip((PointF*)vpt,ninput,(PointF*)cpt,&nclipped);
}


void _init_tshape(Point2f shape[3], float R=100)
{
	shape[0]=Point2f(0, R);
	shape[1]=Point2f(-sqrt(3.0)*R*0.5f, -0.5*R);
	shape[2]=Point2f( sqrt(3.0)*R*0.5f, -0.5*R);
}

void _transform_tshape(const Point2f shape[3], Point2f dshape[3], const cv::Matx33d &T, const Point2f &footPoint)
{
	Point2f center( (shape[0]+shape[1]+shape[2]));
	center.x/=3.0f;
	center.y/=3.0f;

	Point2f shapex[3];
	for(int i=0; i<3; ++i)
	{
		shapex[i]= (shape[i]-center+footPoint);
	}

	Point2f shapey[3];
	do_transform(T,shapex,shapey,3);

	for(int i=0; i<3; ++i)
	{
		dshape[i]=shape[i]+shapey[i]-shapex[i];
	}
}

float _get_shape_error(const Point2f shapex[3], const Point2f shapey[3])
{
	float err=0;
	for(int i=0; i<3; ++i)
	{
		Point2f dv(shapex[i]-shapey[i]);
		err+=dv.dot(dv);
	}
	return sqrt(err)/3.0f;
//	return err;
}

void dp_kernel_v2(const char *bgBuffer, int nbg, int NM, _DPTNode *vA, _DPTNode *vAT, DPNode *vdpCur, DPNode *vdpPrev, DPKernelData *cdata, const Point2f fshape[3], cv::Matx33d fT, cv::Point2f footPoint, float WEIGHT_COVERAGE, float WEIGHT_DISTORTION, float WEIGHT_SMOOTH, float WEIGHT_TRANSFORM, bool IS_DYNAMIC_BG)
{
	cv::Matx33d fTX(fT);
	get_transform<4>(fT,fTX,cdata->m_fgCorners,NULL,false);

	if(!cdata->m_vWeight)
	{
		cdata->m_vWeight=new float[(N_FOOT_NBRS+4)*NM];
		for(int i=0; i<NM; ++i)
		{
			float *wei=cdata->m_vWeight+(N_FOOT_NBRS+4)*i;

			wei[0]=1;
			for(int i=1; i<N_FOOT_NBRS; ++i)
				wei[i]=0.8f;
			for(int i=0; i<4; ++i)
				wei[i+N_FOOT_NBRS]=0.2;	
		}
	}

	for(int bix=0; bix<nbg*NM; ++bix)
	{
		const int dbi=bix/NM;
		const int motion=cdata->m_motion[bix%NM];
		const float *wei=cdata->m_vWeight;//+(N_FOOT_NBRS+4)*motion;

		cv::Matx33d TkInv, Tx;
		
		_BgFrameData *bd=(_BgFrameData*)(bgBuffer+((int*)bgBuffer)[dbi]);

		float min_err=1e10;
		int   father=-1;
		DPError  errs;
#if _SAVE_TK
		cv::Matx33d  mTk;
#endif

		cv::Point2f tpt[N_FOOT_NBRS+ 4], ptx[4], dshape[3];

		const int prevFather=vdpPrev[bix].m_father/NM;

		for(int mi=0; mi<NM; ++mi)
		{
			for(size_t k=0; k<bd->m_nnbr; ++k)
			{
				if(IS_DYNAMIC_BG && !(dbi-bd->m_vnbr[k].m_index>0 && dbi-bd->m_vnbr[k].m_index<=3))
					continue;

				int ni=bd->m_vnbr[k].m_index*NM+mi;

				if(IS_DYNAMIC_BG && dbi==bd->m_vnbr[k].m_index && vdpPrev[ni].m_father/NM==dbi)
					continue;

				{
				//	cv::Matx33d Tk=fT*vA[ni].m_A*cdata->bgT*bd->m_vnbr[k].m_T;
					cv::Matx33d Tk=fT*vA[ni].m_A*bd->m_vnbr[k].m_T;

					if(!_inv_mat33(Tk,TkInv))
						continue;

					float t_error=0;
					if(motion!=MOTION_HOMOGRAPHY)
					{
						do_transform(TkInv,cdata->m_footCorners,footPoint,tpt,N_FOOT_NBRS);

						do_transform(TkInv,cdata->m_fgCorners,Point2f(0,0),tpt+N_FOOT_NBRS,4);

					//	if(!get_transform<N_FOOT_NBRS>(Tk,Tx,tpt,&t_error,motion==MOTION_TSR? true : false))
						if(!get_transform<N_FOOT_NBRS+4,N_FOOT_NBRS+4>(Tk,Tx,tpt,wei,&t_error,motion==MOTION_TSR? true : false))
					//	if(!get_restricted_motion<N_FOOT_NBRS+4, 4>(Tk,Tx,tpt,vA[ni].m_bgTCorners, wei,&t_error,0.0f,motion==MOTION_TSR? true : false))
							continue;

						Tk=Tx;
						t_error=WEIGHT_TRANSFORM*t_error;
					}

					int nclipped=0;
					if(bd->m_npt>0)
					{
						do_transform(Tk,bd->GetPolyPoints(),cdata->m_fgTransformedPolyBuffer,bd->m_npt);
						nclipped=bd->m_npt;
					}
					else
					{
						do_transform(Tk,cdata->m_bgCorners,cdata->m_fgTransformedPolyBuffer,4);
						nclipped=4;
					}

		//			SutherlandHodgman sh(RectF(0,0,cdata->m_fgSize.width,cdata->m_fgSize.height));
					SutherlandHodgman sh(RectF(cdata->m_fgCorners[0].x,cdata->m_fgCorners[0].y,cdata->m_fgCorners[2].x-cdata->m_fgCorners[0].x,cdata->m_fgCorners[2].y-cdata->m_fgCorners[0].y));
					poly_clip(&sh, cdata->m_fgTransformedPolyBuffer,nclipped, cdata->m_fgClippedPolyBuffer,nclipped);

					float c_error=WEIGHT_COVERAGE*_get_coverage(cdata->m_fgClippedPolyBuffer,nclipped,cdata->m_fgArea);

					do_transform(Tk, cdata->m_bgCorners, ptx, 4);

					float s_error=WEIGHT_SMOOTH*_get_smooth_error(ptx,vA[ni].m_bgTCorners);

#if 1
					cv::Matx33d Tb=vA[ni].m_A*/*cdata->bgT**/bd->m_vnbr[k].m_T;
					_inv_mat33(Tb,Tb);
					Tb=Tk*Tb;
				//	s_error=WEIGHT_SMOOTH*_get_matching_error(cdata->m_fgCorners,4,fTX,Tb);
#endif
					for(int i=0; i<N_FOOT_NBRS; ++i)
						tpt[i]=cdata->m_footCorners[i]+footPoint;
					get_transform<N_FOOT_NBRS>(Tb,Tb,tpt,NULL,true);

					_transform_tshape(vdpPrev[ni].m_bshape, dshape, Tb, footPoint);
				//	float a_error=_get_shape_error(dshape, fshape);
					
					float d_error=WEIGHT_DISTORTION*_get_bg_distortion(ptx);

				//	float error=c_error+s_error+d_error+t_error;// + a_error;
					float error=t_error+c_error+d_error+s_error;
					error+=vdpPrev[ni].m_error;

					if(error<min_err)
					{
						min_err=error;
						father=ni;
						vAT[bix].m_A=Tk;//*cdata->bgTInv;
						arr_assign(vAT[bix].m_bgTCorners, ptx);
						arr_assign(vdpCur[bix].m_bshape, dshape);

						errs.m_bgDistortion=d_error;
						errs.m_coverage=c_error;
						errs.m_smooth=s_error;
						errs.m_matching=t_error;
#if _SAVE_TK
						mTk=Tk;
#endif
					}
				}
			}
		}

		vdpCur[bix].m_father=father;
	//	vdpCur[bix].m_error=__max(vdpCur[bix].m_error, min_err);
		vdpCur[bix].m_error=min_err;
		vdpCur[bix].m_errors=errs;
#if _SAVE_TK
		vdpCur[bix].m_Tk=mTk;
#endif
	}
}

void _search_v2(const char *fgBuffer, int nfg, const char *bgBuffer, int nbg, cv::Matx33d bgT, cv::Size bgSize, cv::Size fgSize, FgDataSet &fgDataSet, std::vector<TShape> &shape, CSlippage::DPParam param)
{
	const int NM=param.NM;

	std::vector<std::vector<DPNode> >  vdp(nfg);

	cv::Matx33d mI=cv::Matx33d(1,0,0,0,1,0,0,0,1);

	SutherlandHodgman fgClipper(RectF(0,0,fgSize.width,fgSize.height));

	DPKernelData cdata;
	cdata.bgT=bgT;
	_inv_mat33(cdata.bgT,cdata.bgTInv);
	arr_assign(cdata.m_motion, param.motion);
	
	cdata.m_fgSize=fgSize;
	cdata.m_fgClipper=&fgClipper;
	cdata.m_fgClippedPolyBuffer=new cv::Point2f[1024*2];
	cdata.m_fgTransformedPolyBuffer=cdata.m_fgClippedPolyBuffer+1024;

	{
		cv::Point2f bgCorners[4]={Point2f(0.0f,0.0f), Point2f(0,(float)bgSize.height), Point2f((float)bgSize.width,(float)bgSize.height), Point2f((float)bgSize.width,0) };
		arr_assign(cdata.m_bgCorners, bgCorners);
	}

	{
		int extension_up = param.foreground_extension[0];
		int extension_down = param.foreground_extension[1];
		int extension_left = param.foreground_extension[2];
		int extension_right = param.foreground_extension[3];

		int fg_height = fgSize.height;
		int fg_width = fgSize.width;

		Point2f point_ul = Point2f(-extension_left, -extension_up);
		Point2f point_dl = Point2f(-extension_left, fg_height+extension_down);
		Point2f point_dr = Point2f(fg_width+extension_right, fg_height+extension_down);
		Point2f point_ur = Point2f(fg_width+extension_right, -extension_up);

		cv::Point2f fpt[4]={point_ul, point_dl, point_dr, point_ur};

	//	cv::Point2f fpt[4]={Point2f(0.0f,0.0f), Point2f(0,(float)fgSize.height), Point2f((float)fgSize.width,(float)fgSize.height), Point2f((float)fgSize.width,0) };
	//	cv::Point2f fpt[4]={Point2f(0.0f,0.0f), Point2f((float)fgSize.width,0), Point2f((float)fgSize.width,(float)fgSize.height),  Point2f(0,(float)fgSize.height)};
		arr_assign(cdata.m_fgCorners,fpt);
		cdata.m_fgArea=_poly_area(fpt);
	}

	{
		init_cvt_points(cdata.m_footCorners);
	}

	{
		do_transform(bgT, cdata.m_bgCorners, cdata.m_bgTCorners, 4);
		cdata.m_bgTArea=_poly_area(cdata.m_bgTCorners);
	}

	cv::Point2f fshape[3];
	_init_tshape(fshape);

	ff::AutoArrayPtr<_DPTNode> vA(new _DPTNode[nbg*NM]), vAT(new _DPTNode[nbg*NM]);

	vdp[0].resize(nbg*NM);
	for(size_t i=0; i<nbg*NM; ++i)
	{
		vdp[0][i].m_father=-1;
		vdp[0][i].m_error=((i/NM)>=(size_t)param.firstFrameRange[0] && (i/NM)<(size_t)param.firstFrameRange[1])? 0 : 1e20;
#if _SAVE_TK
		vdp[0][i].m_Tk=bgT;
#endif
		vA[i].m_A= bgT;

		arr_assign(vA[i].m_bgTCorners, cdata.m_bgTCorners);
		arr_assign(vdp[0][i].m_bshape, fshape);
	}

	const int *fgIndex=(int*)fgBuffer;
	const int *bgIndex=(int*)bgBuffer;

	ff::AutoArrayPtr<DPNode> vdpPrev(new DPNode[nbg*NM]), vdpCur(new DPNode[nbg*NM]);
	memcpy(vdpPrev, &vdp[0][0], sizeof(DPNode)*nbg*NM);


	shape.resize(nfg);
	arr_assign(shape[0].m_bg, fshape);
	arr_assign(shape[0].m_fg, fshape);

	Point2f tpt[N_FOOT_NBRS];

	for(size_t fi=1; fi<nfg; ++fi)
	{
		std::cout<<"dp:"<<fi<<endl;

		_FgFrameData *fd=(_FgFrameData*)(fgBuffer+fgIndex[fi]);

		for(int i=0; i<N_FOOT_NBRS; ++i)
			tpt[i]=cdata.m_footCorners[i]+fd->m_FootPoint;

		cv::Matx33d Tb(fd->m_RelativeAffine);
		get_transform<N_FOOT_NBRS>(Tb,Tb,tpt,NULL,true);

	//	_transform_tshape(fshape, fshape, fd->m_RelativeAffine, ((_FgFrameData*)(fgBuffer+fgIndex[fi-1]))->m_FootPoint);
		_transform_tshape(fshape, fshape, Tb, fd->m_FootPoint);
		arr_assign(shape[fi].m_fg, fshape);

		dp_kernel_v2(bgBuffer,nbg,NM,vA,vAT,vdpCur,vdpPrev,&cdata,fshape,fd->m_RelativeAffine,fd->m_FootPoint,param.WEIGHT_COVERAGE, param.WEIGHT_DISTORTION,param.WEIGHT_SMOOTH,param.WEIGHT_TRANSFORM, param.is_dynamic_bg);

		vdp[fi].resize(nbg*NM);
		memcpy(&vdp[fi][0], vdpCur, sizeof(DPNode)*nbg*NM);
		std::swap(vdpCur, vdpPrev);

		std::swap(vA, vAT);
	}

	float err_min=FLT_MAX;
	int imin=0;
	for(size_t i=0; i<nbg*NM; ++i)
	{
		if(vdp[nfg-1][i].m_error<err_min)
		{
			imin=(int)i;
			err_min=vdp[nfg-1][i].m_error;
		}
		printf("\t%.2f",vdp[nfg-1][i].m_error);
	}

	cout << endl << "min sum error is: " << err_min << endl;
//	system("pause");
//	exit(0);

	for(int i=nfg-1; i>=0; --i)
	{
		FgFrameData *fd=fgDataSet.GetAtIndex(i);

		fd->m_Correspondence = imin/param.NM;
		fd->m_TransformMethod=param.motion[imin%param.NM];
 		fd->m_Errors=vdp[i][imin].m_errors;

		arr_assign(shape[i].m_bg, vdp[i][imin].m_bshape);

#if _SAVE_TK
		fd->m_CorrespondenceAffine=cv::Mat(vdp[i][imin].m_Tk);
		fd->m_CorrespondenceHomography=fd->m_CorrespondenceAffine;
#endif

		imin=vdp[i][imin].m_father;

		printf("\n%d, %d",i, imin);
	}
}


void _to_(const cv::Mat &T, double d[3][3])
{
	assert(T.size()==cv::Size(3,3));

	for(int i=0; i<3; ++i)
	{
		for(int j=0; j<3; ++j)
		{
			d[i][j]=*T.ptr<double>(j,i);
		}
	}
}

void _to_(const cv::Point2f &pt, float d[2])
{
	d[0]=pt.x; d[1]=pt.y;
}

void _to_(const FgFrameData &h, _FgFrameData &d)
{
//	_to_(h.m_RelativeAffine, d.m_RelativeAffine);
	d.m_RelativeAffine=h.m_RelativeAffine;
	d.m_FootPoint=h.m_FootPoints.front();
}

void _to_(const BgFrameData::Neighbor &h, _BgNbr &d)
{
//	_to_(h.m_T, d.m_T);
	d.m_T=h.m_T;
	d.m_index=h.m_index;
}

}

void _dp_search(FgDataSet &fg, std::vector<TShape> &shape,  BgDataSet &bg, const cv::Mat &bgT, cv::Size bgSize, cv::Size fgSize, CSlippage::DPParam param)
{
	int fgBufferSize=fg.Size()*(sizeof(int)+sizeof(_FgFrameData));
	
	char *fgBuffer=new char[fgBufferSize];
	
	int *index=(int*)fgBuffer;
	_FgFrameData *fgData=(_FgFrameData*)(fgBuffer+sizeof(int)*fg.Size());
	for(size_t i=0; i<fg.Size(); ++i)
	{
		_to_(*fg.GetAtIndex(i), fgData[i]);

		index[i]=int((char*)&fgData[i]-(char*)fgBuffer);
	}

	int bgBufferSize=sizeof(int)*bg.Size();
	for(size_t i=0; i<bg.Size(); ++i)
	{
		BgFrameData *bgi=bg.GetAtIndex(i);
		bgBufferSize+=sizeof(_BgFrameData);
		if(bgi->m_nbr.size()>1)
			bgBufferSize += bgi->m_nbr.size()*sizeof(_BgNbr);
	}

	char *bgBuffer=new char[bgBufferSize];

	index=(int*)bgBuffer;
	int pos=sizeof(int)*bg.Size();
	
	for(size_t i=0; i<bg.Size(); ++i)
	{
		_BgFrameData *di=(_BgFrameData*)(bgBuffer+pos);
		BgFrameData *bgi=bg.GetAtIndex(i);

		di->m_npt=(int)bgi->m_polyRegion.size();
		if((int)bgi->m_polyRegion.size() == 0)
		{
			di->m_vpt = NULL;
		}
		else
		{
			di->m_vpt=(cv::Point2f*)&bgi->m_polyRegion.front();
		}

		int k=0;
		for(size_t j=0; j<bgi->m_nbr.size(); ++j)
		{
//			if(bgi->m_nbr[j].m_distortion_err<param.MAX_DST_ERROR && bgi->m_nbr[j].m_transform_err<param.MAX_TSF_ERROR)
			if(j<bgi->m_nTemporalNBR || bgi->m_nbr[j].m_transform_err<param.MAX_TSF_ERROR)
//			if(j<bgi->m_nTemporalNBR)
			{
				_to_(bgi->m_nbr[j], di->m_vnbr[k]);
				++k;
			}
		}

		di->m_nnbr=k;
		int isize=sizeof(*di)+(k>1? (k-1)*sizeof(_BgNbr) : 0);
		index[i]=pos;
		pos+=isize;
	}

	_search_v2(fgBuffer,fg.Size(),bgBuffer,bg.Size(),bgT,bgSize,fgSize,fg,shape,param);

	delete[]fgBuffer;
	delete[]bgBuffer;
}

void dp_solve(FgDataSet &fg, BgDataSet &bg, double bgWorkScale, const cv::Mat &_BGT)
{
#if _SAVE_TK==0
	cv::Point2f tpt[N_FOOT_NBRS];
	init_cvt_points(tpt);

	cv::Mat BGWorkT(cv::Matx33d(bgWorkScale,0,0, 0,bgWorkScale,0, 0,0,1));

	Matx33d BGT(_BGT);
	Matx33d BGTInv;
	_inv_mat33(BGT, BGTInv);

	Matx33d preBgAffine;

	for(uint i=0; i<fg.Size(); ++i)
	{
		std::cout << "frame:" << i << endl;

		FgFrameData *fgFd = fg.GetAtIndex(i);

		if(i==0)
		{
			fgFd->m_CorrespondenceHomography=cv::Mat(BGT);
			fgFd->m_CorrespondenceAffine = cv::Mat(BGT);
			preBgAffine=Matx33d(1,0,0, 0,1,0, 0,0,1);
		}
		else
		{
			FgFrameData *preFgFd = fg.GetAtIndex(i-1);
			int correspondence = fgFd->m_Correspondence;
			int method=fgFd->m_TransformMethod;
			int preCorrespondence = preFgFd->m_Correspondence;

			BgFrameData *bgFd = bg.GetAtIndex(correspondence);
			BgFrameData *preBgFd = bg.GetAtIndex(preCorrespondence);

			Matx33d bgT;

			for(size_t j=0; j<bgFd->m_nbr.size(); ++j)
			{
				if(bgFd->m_nbr[j].m_index==preCorrespondence)
				{
					bgT=Matx33d(bgFd->m_nbr[j].m_T);
					break;
				}
			}

			Matx33d affine (fgFd->m_RelativeAffine);

			Matx33d Tk=affine*preBgAffine*BGT*bgT;
			Matx33d TkInv;
			_inv_mat33(Tk,TkInv);

			fgFd->m_CorrespondenceHomography=cv::Mat(Tk);

			if(method!=MOTION_HOMOGRAPHY)
			{
				Point2f  tptx[N_FOOT_NBRS];
				do_transform(TkInv,tpt,fgFd->m_FootPoints.front(),tptx,N_FOOT_NBRS);

				get_transform<N_FOOT_NBRS>(Tk,Tk,tptx,NULL,method==MOTION_TSR? true:false);
			}

	//		cout<<fgFd->m_CorrespondenceAffine<<endl;

			fgFd->m_CorrespondenceAffine=cv::Mat(Tk);

			preBgAffine=Tk*BGTInv;
		}

		fgFd->m_CorrespondenceAffine=fgFd->m_CorrespondenceAffine*BGWorkT;
		fgFd->m_CorrespondenceHomography=fgFd->m_CorrespondenceHomography*BGWorkT;
	}
#else
	cv::Mat BGWorkT(cv::Matx33d(bgWorkScale,0,0, 0,bgWorkScale,0, 0,0,1));

	for(uint i=0; i<fg.Size(); ++i)
	{
		std::cout << "frame:" << i << endl;

		FgFrameData *fgFd = fg.GetAtIndex(i);

		fgFd->m_CorrespondenceAffine=fgFd->m_CorrespondenceAffine*BGWorkT;
		fgFd->m_CorrespondenceHomography=fgFd->m_CorrespondenceHomography*BGWorkT;
	}

#endif
}



