#ifndef PART_MODEL_FUNS_H
#define PART_MODEL_FUNS_H

#include "PartModel_types.h"

void gradMag( float *I, float *M, float *O, int h, int w, int d, bool full );

void fhog( float *M, float *O, float *H, int h, int w, int binSize, int nOrients, int softBin, float clip );

void hog( float *M, float *O, float *H, int h, int w, int binSize,int nOrients, int softBin, bool full, float clip );

//Mat qt(Mat &feat, Mat &C, const int &padx, const int &pady);
Mat qt(Mat &feat, Mat &C1, Mat &C2, const int &padx, const int &pady);

void qtpyra( CSC_MODEL &model, FEATURE_PYRAMID &pyra );

void draw_img( Mat &img, vector<FLOATS_5> &detections, float ElapsedTime );

void		loadCscModel( const string FileName, CSC_MODEL &csc_model );

void		featpyramid( const Mat &im, const CSC_MODEL &model, FEATURE_PYRAMID &pyra );

void		featpyramid2( const Mat &im, const CSC_MODEL &model, FEATURE_PYRAMID &pyra );

Mat filter2lut(Mat &filter, Mat &C);

Mat		features( const Mat &image, const int sbin,const Mat &T1,const Mat&T2 );
//Mat		features( const Mat &image, const int sbin );

Mat		features2( const Mat &image, const int sbin, const int pad[2]);

void		project_pyramid( const CSC_MODEL &model, FEATURE_PYRAMID &pyra );

Mat		project( const Mat &f, const Mat &coeff );

Mat		loc_feat( const CSC_MODEL &model, int num_levels );

void		fconv( const Mat &A, const vector<Mat> &Fs, int St, int Ed, vector<Mat> &Rs );

void		fconv_root_qt(const CSC_MODEL &model,const Mat &A, vector<Mat> &Rs);

int			yuGetCurrentTime( char flag='S' );

//Mat		cascade( const CSC_MODEL &model, const FEATURE_PYRAMID &pyra, 
			//	const vector<vector<Mat> > &rootscores, const int numrootlocs, const int s );

Mat	cascade_qt( const CSC_MODEL &model, const FEATURE_PYRAMID &pyra, 
			const vector<vector<Mat> > &rootscores, const int numrootlocs, const int s );

// draw and show
void	clipboxes( const Mat &img, Mat &ds, Mat &bs );
void	bboxpred_input( const Mat &ds, const Mat &bs, Mat &A, Mat &x1, Mat &y1, Mat &x2, Mat &y2, Mat &w, Mat &h );
void	bboxpred_get( const CSC_MODEL &model, Mat &ds, const Mat &bs );
void	nms( Mat &boxes, float overlap, Mat &parts = Mat() );
Mat	getboxes( const CSC_MODEL &model, const Mat &image, Mat &coords );
void	showboxes( Mat &img_color, const Mat &boxes );

// auxiliary
void	yuSaveMat( string Name, Mat &mm );
void	loadMat( string filename, Mat &A );
void	loadPyramid( const string FileName, FEATURE_PYRAMID &pyra );
void	yuCvPrint( const Mat &A, int chann );

#endif