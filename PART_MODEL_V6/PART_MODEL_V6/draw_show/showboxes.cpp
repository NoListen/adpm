#include "cv.h"
using namespace cv;

// Draw bounding boxes on top of an image.
void	showboxes( Mat &img_color, const Mat &boxes )
{
	static const Scalar TopBoxColor = CV_RGB(255,0,0);
	static	const Scalar PartBoxColor = CV_RGB(0,0,255);

	int numfilters = int( boxes.cols / 4.f );
	for( int i=numfilters-1; i>=0; i-- ){
		Mat x1s = boxes.col( 4*i );
		Mat y1s = boxes.col( 4*i+1 );
		Mat x2s = boxes.col( 4*i+2 );
		Mat y2s = boxes.col( 4*i+3 );
		// draw each object
		for( int k=0; k<x1s.rows; k++ ){
			float	x1 = x1s.at<float>(k);
			float	y1 = y1s.at<float>(k);
			float	x2 = x2s.at<float>(k);
			float	y2 = y2s.at<float>(k);
			if( x1==0 && y1==0 && x2==0 && y2==0 )
				continue;
			Point2f		UL( x1, y1 );
			Point2f		BR( x2, y2 );
			if( i>0 )
				rectangle( img_color, UL, BR, PartBoxColor );
			else
				rectangle( img_color, UL, BR, TopBoxColor );
		}
	}

}