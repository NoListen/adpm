#ifndef PART_MODEL_TYPES_H
#define PART_MODEL_TYPES_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <memory>

#include "cv.h"
#include "highgui.h"

using namespace std;
using namespace cv;

#ifndef	MIN
#define MIN(x,y)	 (x<y?x:y)
#endif

#ifndef MAX
#define MAX(x,y)	 (x>y?x:y)
#endif

#ifndef INT_INF
//#define INT_INF		INT_MAX	// int32 ranges [-2147483648,2147483647]
// the c runtime file <limits> has defined INT_MAX & INT_MIN & FLT_MAX & FLT_MIN
#define INT_INF	(int)1e8
#endif

#ifndef FLOAT_INF
//#define FLOAT_INF	float(3e38/10000) // in fact, float32 ranges (4e38,4e38)
//#define FLOAT_INF	numeric_limits<float>::infinity() // 1.#INF 
#define FLOAT_INF	(float)1e20
#endif

struct INTS_2
{
	int		x[2];
	INTS_2() { memset(x,0,2*sizeof(int)); }
	int	&	operator[] ( int i ) { return	x[i]; }
	void operator=(INTS_2 rhs) { x[0] = rhs.x[0]; x[1] = rhs.x[1]; }
};

struct FLOATS_4
{
	float	x[4];
	FLOATS_4() { memset(x,0,4*sizeof(float)); }
	float	&	operator [] (int i) { return x[i]; }
	float	operator[](int i) const { return x[i]; }
};

struct FLOATS_5
{
	float	x[5];
	FLOATS_5() { memset(x,0,5*sizeof(float)); }
	float	&	operator [] (int i) { return x[i]; }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~

struct CSC_MODEL
{
	string		Class, year, note;
	int			sbin, interval, numblocks, numcomponents;	
	int			maxsize[2], minsize[2];
	float		thresh, threshing;

	Mat		pca_coeff;
	Mat		C2;
	Mat		C1;
	Mat		T1;
	Mat		T2;
	vector<Mat> rootlut;

	struct{
		int			sbin, dim, truncation_dim;
		bool		extra_octave;
		float		bias;
	}features;

	struct	BBOXPRED{
		vector<float>		x1, y1, x2, y2;
	};
	vector<BBOXPRED>	bboxpred;

	struct ROOT_FILTERS{
		int		blocklabel;
		int		size[2];
		Mat	w;
		Mat	wpca;		
	};
	vector<ROOT_FILTERS>	rootfilters;

	struct OFFSETS{
		float	w;
		int		blocklabel;
	};
	vector<OFFSETS>	offsets;

	vector<Mat>	loc_w;

	struct LOC{
		int		blocklabel;
		vector<float>	w;
	};
	vector<LOC>	loc;

	struct COMPONENTS{
		int		rootindex;
		int		offsetindex;
		struct PARTS{
			int		partindex;
			int		defindex;
		};
		vector<PARTS>	parts;
	};
	vector<COMPONENTS>	components;

	struct PART_FILTERS{
		int		blocklabel;
		Mat	w;
		Mat	wpca;	
		Mat wlut;
	};
	vector<PART_FILTERS>		partfilters;

	struct DEFS{
		int				blocklabel;
		INTS_2		anchor;
		FLOATS_4	w;	// 4-params to calculate deformation cost
	};
	vector<DEFS>		defs;

	struct CASCADE{		
		float				thresh;
		vector<vector<int> >	order;
		vector<vector<float> >	t;		
	}cascade;

};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct FEATURE_PYRAMID{
	vector<Mat>	feat;	
	vector<float>	scales;
	vector<bool>	valid_levels;
	int		imsize[2];
	int		num_levels;
	int		padx, pady;
	// add-on
	vector<Mat>	projfeat;
	vector<Mat> qtfeat;
	vector<Mat>	loc_scores;
};

#endif