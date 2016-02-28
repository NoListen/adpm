#include "cv.h"
using namespace cv;
using namespace std;


//I want to change you!!!!!!!!!!!!!!!

//Mat	project( const Mat &f, const Mat &coeff )
////p = project(f, coeff)
////
////project filter f onto PCA eigenvectors (columns of) coeff
//{
//	int		Rows = f.rows, Cols = f.cols, Chns = f.channels();
//	int		Rows2 = coeff.rows, Cols2 = coeff.cols, Chns2 = coeff.channels();
//	if( Chns!=Rows2 || Chns2!=1 ){
//		printf("Wrong in project() : matrix dimensions don\'t match!\n");
//		throw runtime_error("");
//	}
//
//	int		Chnsg = Cols2;
//	Mat	g( Rows, Cols, CV_32FC(Chnsg) );
//	Mat	g2( Rows, Cols*Chnsg, CV_32FC1, (void*)(g.ptr<float>(0,0)) );
//
//	Mat	f2( Rows, Cols*Chns, CV_32FC1, (void*)(f.ptr<float>(0,0)) );
//	for( int i=0; i<Cols; i++ ){
//		Mat	f3 = f2( Rect(i*Chns,0,Chns,Rows) );
//		Mat	g3 = g2( Rect(i*Cols2,0,Chnsg,Rows) );
//		g3 = f3 * coeff;
//	}
//	// make some adjustment
//	//Mat g(Rows,Cols,CV_32FC(Chnsg));
//
//	return	g;
//}
//
//// TEST
////Mat	f;
////loadMat( "f.txt", f );
////Mat	coeff;
////loadMat( "coeff.txt", coeff );
////Mat	g = project( f, coeff );
////yuSaveMat("g",g);

Mat project(const Mat &f, const Mat &coeff)
{
	int fr = f.rows, fc = f.cols, fch = f.channels();
	int cr = coeff.rows, cc = coeff.cols, cch = coeff.channels();
	// I know it is one CChs
	if (fch != cr || cch != 1)
	{
		printf("Matrix dimensions don\'t match! project()\n");
		throw runtime_error("");
	}

	Mat g(fr , fc, CV_32FC(cc));

	//Mat g_2(fr, fc*fch, CV_32FC1, (void*)(g.ptr<float>(0,0)));	
	Mat g_2(fr*fc, cc, CV_32FC1, (void*)(g.ptr<float>(0,0)));

	Mat f_2(fr*fc, fch, CV_32FC1, (void*)(f.ptr<float>(0,0)));

	g_2 = f_2*coeff;

	return g;
}