#include "../PartModel_funs.h"
#include <math.h>
#include "string.h"
#include "sse.hpp"
#include <assert.h>
#include <opencv2/opencv.hpp>

Mat resize(Mat const &src_mat, float const &scale);

struct getFeat_data{
	vector<Mat> *MatArr;
	FEATURE_PYRAMID *pyra;
	int interval, level, len;
	int *pads, *binSz;
};



void * getFeat( void *p )
{
	getFeat_data *D = (getFeat_data*)p;
	int level = D->level;
	int interval = D->interval;
	int *binSz = D->binSz;
	int len = D->len;
	int *pads = D->pads;
	vector<Mat> &feat = D->pyra->feat;
	vector<Mat> *MatArr = D->MatArr;
	//printf("thread-%d begins\n",level);
	for( int i=0; i<len; i++ )
		//feat[level+i] = PM_type::features( (*MatArr)[level+i], binSz[level+i], pads );
		//feat[level+i] = PM_type::features14_2( (*MatArr)[level+i], binSz[level+i], pads );
		feat[level+i] = features2( (*MatArr)[level+i], binSz[level+i], pads );
	//printf("thread-%d ends\n",level);
	return NULL;
}

void	featpyramid2( const Mat &im, const CSC_MODEL &model, FEATURE_PYRAMID &pyra )
//% Compute a feature pyramid.
//%   pyra = featpyramid(im, model, padx, pady)
//%
//% Return value
//%   pyra    Feature pyramid (see details below)
//%
//% Arguments
//%   im      Input image
//%   model   Model (for use in determining amount of 
//				   %           padding if pad{x,y} not given)
//				   %   padx    Amount of padding in the x direction (for each level)
//				   %   pady    Amount of padding in the y direction (for each level)
//				   %
//				   % Pyramid structure (basics)
//				   %   pyra.feat{i}    The i-th level of the feature pyramid
//				   %   pyra.feat{i+interval} 
//%                   Feature map computed at exactly half the 
//%                   resolution of pyra.feat{i}
{
	//ofstream fout("scaled.txt");
	if( im.type()!=CV_32FC3 ) throw runtime_error("");
	// Amount to pad each level of the feature pyramid.
	// We pad the feature maps to detect partially visible objects.
	// padx    Amount to pad in the x direction
	// pady    Amount to pad in the y direction
	// Use the dimensions of the max over detection windows
	int		padx = model.maxsize[1];
	int		pady = model.maxsize[0];
	int		pads[2] = { pady + 1, padx+ 1 };

	// I think the extra_interval is useless for that the DPM is weaker in detecting small objects, extra calc is unnecessary.
	// if( model.features.extra_octave ) extra_interval = model.interval;	
	
	int extra_interval = 0;
	if (model.features.extra_octave)
	{
		extra_interval = model.interval;
	}
	//extra_interval = 0;

	int		sbin = model.sbin;
	int		interval = model.interval;
	float	sc = powf( 2.f, 1.f/interval );
	int		imsize[2] = { im.rows, im.cols };
	int		min_imsize = MIN( imsize[0],imsize[1] ); // min(imsize） 是指长宽里面最短的
	int		max_scale = 1 + int(logf(min_imsize/(5.f*sbin))/logf(sc));
	int		SZ = max_scale + interval + extra_interval;

	pyra.imsize[0] = imsize[0];
	pyra.imsize[1] = imsize[1];
	pyra.feat.resize( SZ );
	pyra.scales.resize( SZ );
	vector<int> BinSz( SZ, sbin );

	// resized images for each level
	static vector<Mat> MatArr;
	MatArr.resize( SZ );
	
	MatArr[0] = im;
	sc = 1.f/sc; float scf = 1.f;
	// 2 1/sqrt(2) 1 sqrt(2) 0.5 0.5/sqrt() 0.25

	


	//for (int k = 0; k < 32; ++k)
	//	for (int i = 0; i < hb; ++i)
	//		for (int j = 0; j < wb; ++j)
	//			H[i*wb+j+hb*wb*k] = H_temp[i+j*hb+hb*wb*k];

	//for (int i = 0;  i < 32*hb*wb; ++i)
	//	fout<<H[i]<<' ';

	//for (int i = 0; i<interval; ++i)
	//{
	//	 resize(MatArr[0], MatArr[i], Size(), scf, scf, INTER_AREA);

	//	if (extra_interval > 0)
	//	{
	//		BinSz[i] = sbin/4;
	//		pyra.scales[i] = 4*scf;
	//		MatArr[i+extra_interval] = MatArr[i];
	//	}

	//	BinSz[i+extra_interval] = sbin/2;
	//	pyra.scales[i+extra_interval] = 2*scf;

	//	//BinSz[i+interval] = sbin;
	//	pyra.scales[i+interval+extra_interval] = scf;
	//	MatArr[i+interval+extra_interval] = MatArr[i];

	//	scf *= sc;

	//	for (int j = i + interval; j < max_scale; j+=interval)
	//	{
	//		resize( MatArr[j+extra_interval],MatArr[j+interval+extra_interval], Size(), 0.5f, 0.5f, INTER_AREA);
	//		pyra.scales[j+interval+extra_interval] = 0.5f * pyra.scales[j+extra_interval];
	//	}

	//}

	for (int i = 0; i<interval; ++i)
	{
		MatArr[i] = resize(MatArr[0],scf);

		if (extra_interval > 0)
		{
			BinSz[i] = sbin/4;
			pyra.scales[i] = 4*scf;
			MatArr[i+extra_interval] = MatArr[i];
		}

		BinSz[i+extra_interval] = sbin/2;
		pyra.scales[i+extra_interval] = 2*scf;

		//BinSz[i+interval] = sbin;
		pyra.scales[i+interval+extra_interval] = scf;
		MatArr[i+interval+extra_interval] = MatArr[i];

		scf *= sc;

		for (int j = i + interval; j < max_scale; j+=interval)
		{
			MatArr[j+interval+extra_interval] = resize(MatArr[j+extra_interval],0.5f);
			pyra.scales[j+interval+extra_interval] = 0.5f * pyra.scales[j+extra_interval];
		}

	}

	for (int i = 0; i < SZ; ++i)
	{
		pyra.feat[i] = features2(MatArr[i],BinSz[i],pads);
	}

	pyra.num_levels = pyra.feat.size();
	pyra.padx = padx;
	pyra.pady = pady;

	return;
}