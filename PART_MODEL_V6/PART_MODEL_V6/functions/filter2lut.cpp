#include "../PartModel_funs.h"
Mat filter2lut(Mat &filter, Mat &C)
{
	Mat lut(filter.rows,filter.cols,CV_32FC(C.rows));
	Mat lut2d(filter.rows*filter.cols,C.rows,CV_32FC1,(void*)(lut.ptr<float>(0,0)));
	Mat filter2d(filter.rows*filter.cols,filter.channels(),CV_32FC1,(void*)(filter.ptr<float>(0,0)));
	
	//ofstream fout("verifyfilter.txt");
	//fout<<filter2d;
	int tableidx = 0;
	for (int i = 0; i < lut.rows; ++i)
		for (int j = 0; j < lut.cols; ++j)
		{
			Mat lut1d = lut2d(Rect(0,tableidx,C.rows,1));
			Mat filter1d = filter2d(Rect(0,tableidx,filter.channels(),1));
			//cout<<filter1d<<endl;
			lut1d = (C*filter1d.t()).t();
			//cout<<lut1d<<endl;
			++tableidx;
		}

	return lut;
}
