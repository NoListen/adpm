/* WARNING: This is an implementation of filter convolutions using SSE instructions. */

#include "cv.h"
#include <xmmintrin.h>
using namespace cv;
using namespace std;

float	SSE_DOT( const Mat &A, const Mat &B )
{
	if( A.channels()!=32 || B.channels()!=32 )
		throw runtime_error("");
	__m128 a,b,c;
	__m128 v = _mm_setzero_ps();
	for( int i=0; i<A.rows; i++ ){
		const float *A_off = A.ptr<float>(i,0);
		const float *B_off = B.ptr<float>(i,0);
		for( int j=0; j<A.cols; j++ ){
			a = _mm_load_ps(A_off+0);
			b = _mm_load_ps(B_off+0);
			c = _mm_mul_ps(a, b);
			v = _mm_add_ps(v, c);

			a = _mm_load_ps(A_off+4);
			b = _mm_load_ps(B_off+4);
			c = _mm_mul_ps(a, b);
			v = _mm_add_ps(v, c);

			a = _mm_load_ps(A_off+8);
			b = _mm_load_ps(B_off+8);
			c = _mm_mul_ps(a, b);
			v = _mm_add_ps(v, c);

			a = _mm_load_ps(A_off+12);
			b = _mm_load_ps(B_off+12);
			c = _mm_mul_ps(a, b);
			v = _mm_add_ps(v, c);

			a = _mm_load_ps(A_off+16);
			b = _mm_load_ps(B_off+16);
			c = _mm_mul_ps(a, b);
			v = _mm_add_ps(v, c);

			a = _mm_load_ps(A_off+20);
			b = _mm_load_ps(B_off+20);
			c = _mm_mul_ps(a, b);
			v = _mm_add_ps(v, c);

			a = _mm_load_ps(A_off+24);
			b = _mm_load_ps(B_off+24);
			c = _mm_mul_ps(a, b);
			v = _mm_add_ps(v, c);

			a = _mm_load_ps(A_off+28);
			b = _mm_load_ps(B_off+28);
			c = _mm_mul_ps(a, b);
			v = _mm_add_ps(v, c);

			A_off += 32;
			B_off += 32;
		}
	}
	float	vals = v.m128_f32[0] + v.m128_f32[1] + v.m128_f32[2] + v.m128_f32[3];
	return	vals;
}

void	fconv_SSE( const Mat &A, const Mat &F, Mat &R )
{
	int NUM_FEATURES = 32;
	int RowA = A.rows, ColA = A.cols, NumFeatures = A.channels();
	int RowF = F.rows, ColF = F.cols, ChnF = F.channels();
	if( NumFeatures!=ChnF || NumFeatures!=NUM_FEATURES )
		throw runtime_error("");

	int RowR = RowA - RowF + 1, ColR = ColA - ColF + 1;
	R.create( RowR, ColR, CV_32FC1 );

	const float *F_src = F.ptr<float>(0,0);
	const float *A_src0 = A.ptr<float>(0,0);
	float *R_src0 = R.ptr<float>(0,0);

	__m128 a,b,c;
	for( int rr=0; rr<RowR; rr++ ){
		const float *A_src1 = A_src0 + rr*A.cols*NUM_FEATURES; // start addr of A.row(rr)
		float *R_scr1 = R_src0 + rr*R.cols; // start addr of R.row(rr)
		for( int cc=0; cc<ColR; cc++ ){
			const float *A_src= A_src1 + cc*NUM_FEATURES;// A.ptr<float>(rr,cc);
			float *R_src = R_scr1 + cc;

			// THE acceleration trick of using SSE programming >>>
			__m128 v = _mm_setzero_ps();
			for( int rp=0; rp<RowF; rp++ ){
				const float *A_off = A_src + rp*A.cols*NUM_FEATURES;
				const float *B_off = F_src + rp*F.cols*NUM_FEATURES;
				for( int cp=0; cp<ColF; cp++ ){
					//
					a = _mm_load_ps(A_off+0);
					b = _mm_load_ps(B_off+0);
					c = _mm_mul_ps(a, b);
					v = _mm_add_ps(v, c);
					//
					a = _mm_load_ps(A_off+4);
					b = _mm_load_ps(B_off+4);
					c = _mm_mul_ps(a, b);
					v = _mm_add_ps(v, c);
					//
					a = _mm_load_ps(A_off+8);
					b = _mm_load_ps(B_off+8);
					c = _mm_mul_ps(a, b);
					v = _mm_add_ps(v, c);
					//
					a = _mm_load_ps(A_off+12);
					b = _mm_load_ps(B_off+12);
					c = _mm_mul_ps(a, b);
					v = _mm_add_ps(v, c);
					//
					a = _mm_load_ps(A_off+16);
					b = _mm_load_ps(B_off+16);
					c = _mm_mul_ps(a, b);
					v = _mm_add_ps(v, c);
					//
					a = _mm_load_ps(A_off+20);
					b = _mm_load_ps(B_off+20);
					c = _mm_mul_ps(a, b);
					v = _mm_add_ps(v, c);
					//
					a = _mm_load_ps(A_off+24);
					b = _mm_load_ps(B_off+24);
					c = _mm_mul_ps(a, b);
					v = _mm_add_ps(v, c);
					//
					a = _mm_load_ps(A_off+28);
					b = _mm_load_ps(B_off+28);
					c = _mm_mul_ps(a, b);
					v = _mm_add_ps(v, c);
					//
					A_off += NUM_FEATURES;
					B_off += NUM_FEATURES;
				}
			}
			// <<< END of SSE programming

			*R_src =	 v.m128_f32[0] + v.m128_f32[1] + v.m128_f32[2] + v.m128_f32[3];
		}
	}
}