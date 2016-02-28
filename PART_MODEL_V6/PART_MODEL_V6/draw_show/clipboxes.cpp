#include "cv.h"
using namespace cv;

// Clip detection windows to image the boundary.
void	clipboxes( const Mat &img, Mat &ds, Mat &bs )
{
	if( !ds.empty() ){
		Mat	ds_x1 = ds.col(0);
		Mat	ds_y1 = ds.col(1);
		Mat	ds_x2 = ds.col(2);
		Mat	ds_y2 = ds.col(3);
		ds_x1 = cv::max( ds_x1, 0 );
		ds_y1 = cv::max( ds_y1, 0 );
		ds_x2 = cv::min( ds_x2, img.cols-1 );
		ds_y2 = cv::min( ds_y2, img.rows-1 );
		// remove invalid detections
		Mat	w = ds_x2 - ds_x1;
		Mat	h = ds_y2 - ds_y1;
		Mat	invalids = ( (w<1) | (h<1) );
		// 
		int		cnt = countNonZero( invalids );
		Mat	ds_cpy( ds.rows-cnt, ds.cols, ds.type() );
		int		j = 0;
		for( int i=0; i!=ds.rows; i++ ){
			if( invalids.at<uchar>(i) )
				continue;
			ds.row(i).copyTo( ds_cpy.row(j) );
			j++;
		}
		ds.create( ds_cpy.rows, ds_cpy.cols, ds_cpy.type() );
		ds_cpy.copyTo(ds);
		//
		if( !bs.empty() ){
			Mat	bs_cpy( bs.rows-cnt, bs.cols, bs.type() );
			int		j = 0;
			for( int i=0; i!=bs.rows; i++ ){
				if( invalids.at<uchar>(i) )
					continue;
				bs.row(i).copyTo( bs_cpy.row(j) );
				j++;
			}
			bs.create( bs_cpy.rows, bs_cpy.cols, bs_cpy.type() );
			bs_cpy.copyTo(bs);
		}
	}
}