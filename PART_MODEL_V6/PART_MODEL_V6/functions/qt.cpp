#include "../PartModel_funs.h"

static inline float min(float x, float y) { return (x <= y ? x : y); }

static inline float sq(float x) {return x*x;}

static inline float absq(float x) {return x>0?x:-x;}
Mat qt(Mat &feat, Mat &C1,Mat &C2 ,const int &padx, const int &pady)
{
	//Mat qtfeat = Mat::zeros(feat.rows,feat.cols,CV_32SC1);
	Mat qtfeat (feat.rows,feat.cols,CV_32SC1,cv::Scalar(144));

	Mat feat2d(feat.rows*feat.cols,feat.channels(),CV_32FC1,(void*)feat.ptr<float>(0,0));
	//ofstream fop("feat.txt");

	int class_c1 = C1.rows;
	int class_c2 = C2.rows/class_c1;
	//cout<<class_c1<<' '<<class_c2<<endl;

	for (int y = padx; y <= feat.rows-padx; ++y)
		for (int x = pady; x <= feat.cols-pady; ++x)
		{
			Mat featv = feat2d(Rect(0,y*feat.cols+x,32,1));
			//cout<<featv;
		//	cout<<endl;
			double dis = 100000000;
			float tmp;
			int q1 = 0,q2 = 0;
			float *C1_pt = C1.ptr<float>(0,0);
			for (int i = 0 ; i < class_c1; ++i)
			{
				tmp = 0;
				float * featv_pt = featv.ptr<float>(0,0);
				for (int j = 0 ; j < 32; ++j)
				{
					tmp += absq(*C1_pt - *(featv_pt+j));
					++C1_pt;
				}

				if (tmp < dis)
				{
					q1 = i;
					dis = tmp;
				}

			}
			dis = 100000000;
			float *C2_pt = C2.ptr<float>(q1*16,0);
			if (q1 != 9)
			{
				for (int i = 0; i < class_c2; ++i)
				{
					tmp = 0;
				
					float * featv_pt = featv.ptr<float>(0,0);
					for (int j = 0 ; j < 32; ++j)
					{
						tmp += sq(*C2_pt - *(featv_pt+j));
						++C2_pt;
					}

					if (tmp < dis)
					{
						q2 = i;
						dis = tmp;
					}
				}
			}
			qtfeat.at<int>(y,x) = q1*16+q2;
		}
	//fop<<qtfeat;
	return qtfeat;
}