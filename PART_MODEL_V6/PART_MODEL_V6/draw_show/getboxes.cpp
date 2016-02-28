#include "../PartModel_funs.h"

Mat	getboxes( const CSC_MODEL &model, const Mat &image, Mat &coords )
{
	Mat	b;
	if( coords.empty() )
		return	b;

	Mat dets( coords.rows, 6, coords.type() ); // dets = boxes(:,[1:4 end-1 end]);
	( coords.colRange(0,4) ).copyTo( dets.colRange(0,4) );
	( coords.colRange(coords.cols-2,coords.cols) ).copyTo( dets.colRange(4,6) );
	

	clipboxes( image, dets, coords );

	bboxpred_get( model, dets, coords );

	clipboxes( image, dets, coords );

	nms( dets, 0.5f, coords );
	
	return	dets;
}