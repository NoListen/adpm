#include "../PartModel_funs.h"

void	bboxpred_get( const CSC_MODEL &model, Mat &ds, const Mat &bs )
//% Get predicted bounding boxes.
//%   [bbox, bs_out] = bboxpred_get(bboxpred, ds, bs)
//%
//% Return values
//%   ds_pred   Output detection windows
//%   bs_pred   Output filter bounding boxes
//%
//% Arguments
//%   bboxpred  Bounding box prediction coefficients (see bboxpred_train.m)
//%   ds        Source detection windows
//%   bs        Source filter bounding boxes
{
	Mat	bs_col = bs.col( bs.cols-2 );
	double	maxc = 0;	// number of components
	minMaxLoc( bs_col, NULL, &maxc );
	Mat	bs_int;
	bs_col.convertTo( bs_int, CV_32SC1 );

	//std::set<int>	preds;
	//for( int i=0; i!=bs_int.rows; i++ )
	//	preds.insert( bs_int.at<int>(i) );
	//int		pred_size = preds.size() * 5;
	Mat	ds_pred( ds.rows, 5, ds.type() );
	Mat	bs_pred( bs.rows, 5, bs.type() );
	int		pred_id = 0;

	for( int c=1; c<=(int)maxc+1; c++ ){
		// limit boxes to just component c
		vector<int>		cinds;
		for( int i=0; i<bs.rows; i++ )
			if( bs_int.at<int>(i)==c-1 )
				cinds.push_back(i);
		if( cinds.empty() )
			continue;

		Mat	b( cinds.size(), bs.cols, bs.type() );
		for(unsigned  int i=0; i<cinds.size(); i++ )
			bs.row( cinds[i] ).copyTo( b.row(i) );

		Mat	d( cinds.size(), ds.cols, ds.type() );
		for(unsigned  int i=0; i<cinds.size(); i++ )
			ds.row( cinds[i] ).copyTo( d.row(i) );

		// build test data
		Mat	A;
		Mat	x1, y1, x2, y2, w, h;
		bboxpred_input( d, b.colRange(0,b.cols-2), A, x1, y1, x2, y2, w, h );

		// predict displacements
		const vector<float>	&pred_x1 = model.bboxpred[c-1].x1;
		const vector<float>	&pred_y1 = model.bboxpred[c-1].y1;
		const vector<float>	&pred_x2 = model.bboxpred[c-1].x2;
		const vector<float>	&pred_y2 = model.bboxpred[c-1].y2;
		const Mat	pred_x1_mat( pred_x1.size(), 1, CV_32FC1, const_cast<float*>(&pred_x1[0]) );
		const Mat	pred_y1_mat( pred_y1.size(), 1, CV_32FC1, const_cast<float*>(&pred_y1[0]) );
		const Mat	pred_x2_mat( pred_x2.size(), 1, CV_32FC1, const_cast<float*>(&pred_x2[0]) );
		const Mat	pred_y2_mat( pred_y2.size(), 1, CV_32FC1, const_cast<float*>(&pred_y2[0]) );
		Mat	dx1 = A * pred_x1_mat;
		Mat	dy1 = A * pred_y1_mat;
		Mat	dx2 = A * pred_x2_mat;
		Mat	dy2 = A * pred_y2_mat;

		// compute object location from predicted displacements
		Mat	ds_pred_sub = ds_pred.rowRange(pred_id,pred_id+cinds.size());
		pred_id += cinds.size();
		if( dx1.rows!=ds_pred_sub.rows )
			throw	runtime_error("");
		ds_pred_sub.col( 0 ) = x1 + w.mul(dx1);
		ds_pred_sub.col( 1 ) = y1 + h.mul(dy1);
		ds_pred_sub.col( 2 ) = x2 + w.mul(dx2);
		ds_pred_sub.col( 3 ) = y2 + h.mul(dy2);
		b.col(b.cols-1).copyTo( ds_pred_sub.col( 4 ) );

	}

	ds.create( ds_pred.rows, ds_pred.cols, ds_pred.type() );
	ds_pred.copyTo(ds);

}