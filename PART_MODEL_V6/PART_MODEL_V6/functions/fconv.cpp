#include "../PartModel_types.h"

void	fconv_1( const Mat &A, const Mat &F, Mat &R );

void	fconv( const Mat &A, const vector<Mat> &Fs, int St, int Ed, vector<Mat> &Rs )
// A is feature map, Fs is an array of filters
// Fs(St:Ed-1) will be used separately to A to get respective responses (of each terminal)
// The responses will be stored in Rs
{
	int		len = Ed - St;
	if( len<1 )
		throw runtime_error("");
	Rs.resize( len );
	//ofstream fp("x.txt");
	#pragma omp parallel for
	for( int i=St; i<Ed; i++ ){
		int		j = i - St;
		fconv_1( A, Fs[i], Rs[j] );
		//fp<<Rs[j];
	}
}


void	fconv_1( const Mat &A, const Mat &F, Mat &R )
{
	int		RowA = A.rows, ColA = A.cols, NumFeatures = A.channels();
	int		RowF = F.rows, ColF = F.cols, ChnF = F.channels();
	if( NumFeatures!=ChnF )
		throw runtime_error("");

	int		RowR = RowA - RowF + 1, ColR = ColA - ColF + 1;
	R.create( RowR, ColR, CV_32FC1 );

	for( int r=0; r!=RowR; r++ ){
		for( int c=0; c!=ColR; c++ ){
			Mat	Asub = A( Rect(c,r,ColF,RowF) );
			R.at<float>(r,c) = (float)( F.dot( Asub ) );
		}
	}
}


