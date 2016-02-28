#include <string>
#include <iostream>
#include <fstream>
#include <Windows.h>
#include "cv.h"
using namespace std;
using namespace cv;

void		yuCvPermutate( Mat &f, int *new_row_idx, int *new_col_idx, int *new_dim_idx )
{
	int		Rows = f.rows, Cols = f.cols, Chns = f.channels();

	// task can be down in 3 steps:

	// 1st step: f = f(:,:,new_dim_idx);	
	if( new_dim_idx!=NULL ){
		int		*from_to = new int [Chns*2]();
		for( int i=0; i!=Chns; i++ ){
			from_to[2*i] = new_dim_idx[i];
			from_to[2*i+1] = i;
		}
		Mat		g = f.clone();
		mixChannels( &g, 1, &f, 1, from_to, Chns );
		g.release();
	}	

	// 2nd step: f = f(:,new_col_idx,:);
	if( new_col_idx!=NULL ){
		Mat		f2( Rows, Cols*Chns, CV_MAKETYPE(f.depth(),1), f.data );
		Mat		g2 = f2.clone();
		for( int i=0; i!=Cols; i++ ){
			int			f_col_start = i * Chns;
			int			f_col_end = f_col_start + Chns;
			int			g_col_start = new_col_idx[i] * Chns;
			int			g_col_end = g_col_start + Chns;
			Mat		f_col_block = f2.colRange( f_col_start, f_col_end );
			Mat		g_col_block = g2.colRange( g_col_start, g_col_end );
			g_col_block.copyTo( f_col_block );
		}
		g2.release();
	}

	// 3rd step: f = f(new_row_idx,:,:);
	if( new_row_idx!=NULL ){
		Mat	g3 = f.clone();
		for( int i=0; i!=Rows; i++ ){
			Mat	f_row = f.row( i );
			Mat	g_row = g3.row( new_row_idx[i] );
			g_row.copyTo( f_row );
		}
		g3.release();
	}

}



template <typename T>
void	yuCvPrint( const Mat &A, T type, int chann )
{
	int		Rows = A.rows, Cols = A.cols, Chns = A.channels();
	int		A_ROW = Cols * Chns;
	if( chann>=0 && chann<Chns )
		A_ROW = Cols;	// 只打印特定通道
	else{
		Chns = 1; // 打印所有通道
		chann = 0;
	}
	for( int i=0; i!=Rows; i++ ){
		cout<<"\n------------------"<<i<<"th row-------------------------"<<endl;
		T		*ptr = (T*)(A.data) + i*A.step1(0);
		for( int j=0; j!=A_ROW; j++ )
			cout<<(float)ptr[j*Chns+chann]<<" ";		
	}
	cout<<endl;
}

void	yuCvPrint( const Mat &A, int chann )
{
	int		Depth = A.depth();
	switch( Depth ){
		case CV_8U:
			yuCvPrint( A, (uchar)0, chann );
			break;
		case CV_32S:
			yuCvPrint( A, (int)0, chann );
			break;
		case CV_32F:
			yuCvPrint( A, (float)0, chann );
			break;
		case CV_64F:
			yuCvPrint( A, (double)0, chann );
			break;
		default:
			cout<<"Not supported data type in yuCvLinearize() !"<<endl;
			throw runtime_error("");
	}			
}

void	yuSaveMat( string Name, Mat &mm )
{
	string	Name2 = Name + ".xml";
	FileStorage	fs(Name2,FileStorage::WRITE);
	fs<<Name<<mm;
	fs.release();
}

void	yuInd2Sub( const int Ind, const int *Sz, int &Row, int &Col, int &Dim )
{
	int		Area = Sz[0] * Sz[1];
	Dim = Ind / Area;
	int		Res = Ind - Dim * Area;
	Col = Res / Sz[0];
	Row = Res - Col*Sz[0];
}

string		yuGetCurrentTime()
{
	SYSTEMTIME	system;
	GetLocalTime( &system );
	char	t[20] = { 0 };
	sprintf_s( t, "%d_%d_%d_%d.txt", system.wMonth, system.wDay, system.wHour, system.wMinute );
	string	ret(t);
	return	ret;
}

int		yuGetCurrentTime( char flag )
{
	SYSTEMTIME	system;
	GetLocalTime( &system );

	int		seconds = system.wHour*3600 + system.wMinute*60 + system.wSecond;
	if( flag=='S' )		
		return	seconds;

	int 	Mseconds = 1000*seconds + system.wMilliseconds;
	return	Mseconds;
}	

void	loadMat( string filename, Mat &A )
// 加载MATLAB数据矩阵
{
	ifstream	fs(filename.c_str());
	if( !fs )
		throw runtime_error("");

	int		Rows, Cols, Chns;
	Rows = Cols = Chns = 0;
	fs>>Rows>>Cols>>Chns;
	if( Rows<1 || Cols<1 || Chns<1 )
		throw runtime_error("");
	A.create( Rows, Cols, CV_32FC(Chns) );

	for( int r=0; r!=Rows; r++ ){
		for( int c=0; c!=Cols; c++ ){
			float	*pt = A.ptr<float>(r,c);
			for( int d=0; d!=Chns; d++ )
				fs>>pt[d];
		}
	}

	string		flag;
	fs>>flag;
	if( flag!="END" )
		throw runtime_error("");

}