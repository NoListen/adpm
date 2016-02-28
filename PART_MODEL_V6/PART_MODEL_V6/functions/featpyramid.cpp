#include "../PartModel_funs.h"

void	featpyramid( const Mat &im, const CSC_MODEL &model, FEATURE_PYRAMID &pyra )
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
	// Amount to pad each level of the feature pyramid.
	// We pad the feature maps to detect partially visible objects.
	// padx    Amount to pad in the x direction
	// pady    Amount to pad in the y direction
	// Use the dimensions of the max over detection windows
	int		padx = model.maxsize[1];
	int		pady = model.maxsize[0];

	int		extra_interval = 0;
	if( model.features.extra_octave )
		extra_interval = model.interval;

	int		sbin = model.sbin;
	int		interval = model.interval;
	float	sc = powf( 2.f, 1.f/interval );
	int		imsize[2] = { im.rows, im.cols };
	int		min_imsize = MIN( imsize[0],imsize[1] );
	int		max_scale = 1 + int(logf(min_imsize/(5.f*sbin))/logf(sc));

	int		SZ = max_scale + extra_interval + interval;
	pyra.feat.resize( SZ );
	pyra.scales.resize( SZ, 0 );
	pyra.imsize[0] = imsize[0];
	pyra.imsize[1] = imsize[1];

	Mat	im_float;
	if( im.channels()!=3 )
		throw runtime_error("");
	else if( im.type()==CV_32FC3 )
		im_float = im;
	else
		im.convertTo( im_float, CV_32FC3 );

	for( int i=0; i!=interval; i++ ){
		Mat	scaled;
		float	scale_factor = 1.f / powf(sc,float(i));
		resize( im_float, scaled, Size(), scale_factor, scale_factor, INTER_AREA );
		if( extra_interval>0 ){
			// Optional (sbin/4) x (sbin/4) features
			pyra.feat[i] = features( scaled, sbin/4 ,model.T1,model.T2);
			pyra.scales[i] = 4 * scale_factor;
		}
		// (sbin/2) x (sbin/2) features
		pyra.feat[i+extra_interval] = features( scaled, sbin/2 ,model.T1,model.T2);
		pyra.scales[i+extra_interval] = 2 * scale_factor;
		// sbin x sbin HOG features 
		pyra.feat[i+extra_interval+interval] = features( scaled, sbin ,model.T1,model.T2);
		pyra.scales[i+extra_interval+interval] = scale_factor;
		// Remaining pyramid octaves 
		for( int j=i+interval; j<max_scale; j+=interval ){			
			resize( scaled, scaled, Size(), 0.5, 0.5, INTER_AREA );
			pyra.feat[j+extra_interval+interval] = features( scaled, sbin ,model.T1,model.T2);
			pyra.scales[j+extra_interval+interval] = 0.5f * pyra.scales[j+extra_interval];
		}
	}

	pyra.num_levels = pyra.feat.size();

	int		td = model.features.truncation_dim - 1;
	int		pad_top = pady + 1, pad_left = padx+ 1;
	int		pad_bottom = pad_top, pad_right = pad_left;
	int		Chns = pyra.feat[0].channels();	
	vector<Mat>	feat_plane( Chns );
	vector<Mat>	padded_plane( Chns );
	for( int i=0; i!=pyra.num_levels; i++ ){
		split( pyra.feat[i], feat_plane );
		for( int d=0; d!=Chns; d++ ){
			if( d!=td )
				copyMakeBorder( feat_plane[d], padded_plane[d], pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, Scalar(0) );
			else
				copyMakeBorder( feat_plane[d], padded_plane[d], pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, Scalar(1) );
		}
		int		Rows = pyra.feat[i].rows + pad_top + pad_bottom;
		int		Cols = pyra.feat[i].cols + pad_left + pad_right;
		pyra.feat[i].create( Rows, Cols, CV_32FC(Chns) );
		merge( &padded_plane[0], padded_plane.size(), pyra.feat[i] );
	}

	pyra.valid_levels.resize( pyra.num_levels, true );
	pyra.padx = padx;
	pyra.pady = pady;
}


//	TEST
// 
//MODEL	modelt;
//loadModel( "featpyra_model.txt", modelt);
//Mat		img = imread( "featpyra.bmp" );
//FEATURE_PYRAMID		pyramid;

//featpyramid( img, modelt, pyramid );

//FileStorage file_xml("pyramid.xml", FileStorage::WRITE);
//file_xml<<"pyramid"<<pyramid.feat[5];
//file_xml.release();