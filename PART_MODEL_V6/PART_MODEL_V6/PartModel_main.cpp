#include "PM_dem.h"

//vector<string>	readImgDir( const string DirName, bool createRst = false, vector<string> &ResultDir = vector<string>() ); 
vector<string>  yuStdDirFiles( string DirName, vector<string> FileExtensions );

int	main()
{
	/* I. Two ways to get images */

	string		img_dir = "F://data_seq//singer4";
	//string		img_dir = "C://Users//NListen//Desktop//INRIAPerson//96X160H96//Train//pos";
	//string		img_dir = "C://Users//NListen//Desktop//INRIAPerson//Train//pos";
	//vector<string>	resultnames;
	//vector<string>	imgnames = readImgDir( img_dir_name, true, resultnames );	
	string extensions[] = { ".jpg",".JPG",".png" };
	vector<string>	img_extensions( extensions, extensions+3 );
	vector<string>	imgnames = yuStdDirFiles( img_dir, img_extensions );

	/* II. Perform Part_Model based detection */

	PM_dem	PM( "csc_inria_model.txt" );
	//PM_dem PM("csc_car_model.txt");

	for( int i=0; i<imgnames.size(); i++ ){
		string	img_name = imgnames[i];
		Mat	img_uint8 = imread( img_name.c_str() );	
		if( img_uint8.empty() ){
			cout<<"Cannot get image "<<img_name<<endl;
			getchar();
			return -2;
		}
		cout<<"Processing \""<<img_name<<"\""<<endl;
		Mat	img = PM.prepareImg( img_uint8 );

		string name = "person";
		int tnum = i+1;
		string mid = "";
		while (tnum>0)
		{
			char t = ('0' + (tnum%10));
			mid = t + mid;
			tnum = tnum/10;
		}
		name = name+mid+".jpg";

		PM.detect( img, -0.5f, true, true, name);		
		cout<<"Done with PM\n================================"<<endl;
		char key = waitKey(0);
		if( key==27 )
			break;
	}

	return	0;
}
