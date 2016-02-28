#include "../PartModel_funs.h"

void fconv_root_lut(const Mat &A, const Mat &L, Mat &R);

void fconv_root_qt(const CSC_MODEL &model, const Mat &A, vector<Mat> &Rs)
{
	int len = (model.rootlut).size();
	if (len < 1)
		throw runtime_error("");
	Rs.resize(len);
	//ofstream fp("z.txt");
	#pragma omp parallel for
	for( int i=0; i<len; i++ ){
		fconv_root_lut( A, model.rootlut[i] , Rs[i] );
		//fp<<Rs[i];
		//fp<<"########################################"<<endl;
	}
}

void fconv_root_lut(const Mat &A, const Mat &L, Mat &R)
{
	int RowA = A.rows, ColA = A.cols, NumFeatures = A.channels();
	int RowL = L.rows, ColL = L.cols;

	if (NumFeatures != 1)
		throw runtime_error("");

	int RowR = RowA - RowL + 1, ColR = ColA - ColL + 1;

	R.create(RowR, ColR, CV_32FC1);


	for (int r = 0; r != RowR; ++r)
		for (int c = 0; c != ColR; ++c)
		{
			Mat Asub = A(Rect(c,r,ColL,RowL));
			float sum = 0;
			for (int i = 0 ; i < RowL ; ++i)
				for (int j = 0; j < ColL ; ++j)
				{
					int q = Asub.at<int>(i,j);
					const float * t = L.ptr<float>(i,j);
					sum += *(t+q);
				}
			R.at<float>(r,c) = sum;
		}
}
