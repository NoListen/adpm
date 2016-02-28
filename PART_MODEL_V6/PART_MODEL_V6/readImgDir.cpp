#define HAVE_BOOST

//////////////////////////////////////////////////////////////////////////
#ifdef HAVE_BOOST

#include <string>
#include <vector>
#include <set>
#include <iostream>
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem;

#ifndef	BOOST_FILESYSTEM_NO_DEPRECATED
#define	BOOST_FILESYSTEM_NO_DEPRECATED
#endif

vector<string>	readImgDir( const string DirName, bool createRst = false, vector<string> &ResultDir = vector<string>() ); 
// e.g. DirName = "d:\\My Documents\\test_photos_LCX"

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 读取给定目录下所有图像文件的名称
vector<string>	readImgDir( const string DirName, bool createRst, vector<string> &ResultDir )
{
	path	DirPath( DirName );
	if( DirName.empty() )
		DirPath = current_path();
	if( !is_directory(DirPath) ){
		cout<<"No such directory :"<<endl;
		cout<<DirPath.generic_string()<<endl;
		throw	runtime_error("");
	}
	
	if( createRst ){
		string	Result = DirName + "/result";
		create_directory( Result );
	}

	string		Image_Extensions0[] = { ".jpg", ".bmp", ".tiff", ".png", ".jpeg" };
	set<string>	Image_Extensions( Image_Extensions0, Image_Extensions0+5 );

	vector<string>		img_names, img_file_names;
	directory_iterator	it( DirPath );

	cout<<"Read image names from "<<DirPath<<endl;
	while( it!=directory_iterator() ){
		directory_entry	entity = *(it++);
		path		entity_path = entity.path();
		
		const string ExtendName = entity_path.extension().generic_string();
		if( Image_Extensions.find(ExtendName)==Image_Extensions.end() )
			continue;

		const string	name = entity_path.generic_string();
		img_names.push_back( name );				
		
		//cout<<entity_path.filename()<<"  ";
		string	image_name = entity_path.filename().generic_string();
		img_file_names.push_back( image_name );

		if( createRst ){			
			string result_name( name.begin(), name.end()-image_name.size() );
			result_name = result_name + "result/PM_" + image_name;
			ResultDir.push_back( result_name );
		}
	}	

	//
	cout<<"Have found "<<img_names.size()<<" images : ";
	int		min_num = img_file_names.size();
	min_num = min_num>5 ? 5 : min_num;
	for( int i=0; i<min_num; i++ )
		cout<<img_file_names[i]<<"  ";
	cout<<" ..."<<endl;
	cout<<"Done"<<endl;

	return	img_names;
}

#endif