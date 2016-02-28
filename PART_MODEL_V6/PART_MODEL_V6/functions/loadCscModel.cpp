#include "../PartModel_types.h"

inline
void		yuCheck( bool val )
{
	if( !val ){
		printf("Check failure!\n");
		throw runtime_error("");
	}
}

inline 
void		yuCheck( string a, string b )
{
	if( a!=b ){
		printf("Check failure: \"%s\" does not match to \"%s\" !\n",a,b);
		throw	runtime_error("");
	}
}

inline
ifstream&		operator>>( ifstream &ifs, int &val )
{
	// infinite numbers can be written into file with "inf" markup( both in matlab and in c++ ),
	// but when reading from the file, the "inf" markup is not recognized as a number
	string	tmpS;
	ifs>>tmpS;
	if( tmpS=="Inf" || tmpS=="inf" )
		val = INT_INF;
	else if( tmpS=="-Inf" || tmpS=="-inf" )
		val = -INT_INF;
	else
		val = (int)atoi( tmpS.c_str() );

	return	ifs;	
}

inline
ifstream&		operator>>( ifstream &ifs, float &val )
{
	string	tmpS;
	ifs>>tmpS;
	if( tmpS=="Inf" || tmpS=="inf" )
		val = FLOAT_INF;
	else if( tmpS=="-Inf" || tmpS=="-inf" )
		val = -FLOAT_INF;
	else
		val = (float)atof( tmpS.c_str() );

	return	ifs;
}

void		loadCscModel( const string FileName, CSC_MODEL &csc_model )
// 从文件中加载cascade model
{
	ifstream	MF( FileName.c_str() );
	if( !MF ){
		cout<<"Cannot open model file!"<<endl;
		throw runtime_error("");
	}

	string	Title;
	int		Sz0, Sz, tmp, M, N, D;
	float	*pt;

	// class
	MF>>Title;
	yuCheck(Title,"class");
	MF>>csc_model.Class;
	// year
	MF>>Title;
	yuCheck(Title,"year");
	MF>>csc_model.year;
	// note
	MF>>Title;
	yuCheck(Title,"note");
	MF>>csc_model.note;
	// sbin
	MF>>Title;
	yuCheck(Title,"sbin");
	MF>>csc_model.sbin;
	// interval
	MF>>Title;
	yuCheck(Title,"interval");
	MF>>csc_model.interval;
	// numblocks
	MF>>Title;
	yuCheck(Title,"numblocks");
	MF>>csc_model.numblocks;
	// numcomponents
	MF>>Title;
	yuCheck(Title,"numcomponents");
	MF>>csc_model.numcomponents;
	// maxsize
	MF>>Title;
	yuCheck(Title,"maxsize");
	MF>>csc_model.maxsize[0]>>csc_model.maxsize[1];
	// minsize
	MF>>Title;
	yuCheck(Title,"minsize");
	MF>>csc_model.minsize[0]>>csc_model.minsize[1];
	// thresh
	MF>>Title;
	yuCheck(Title,"thresh");
	MF>>csc_model.thresh;
	
	// pca_coeff
	MF>>Title;
	yuCheck(Title,"pca_coeff");
	MF>>M>>N>>D;
	csc_model.pca_coeff.create( M, N, CV_32FC(D) );
	pt = csc_model.pca_coeff.ptr<float>(0,0);
	Sz = M * N * D;
	tmp = 0;
	while( tmp<Sz )
		MF>>pt[tmp++];

	// features
	MF>>Title;
	yuCheck(Title,"features");
	{
		// sbin
		MF>>Title;
		yuCheck(Title,"sbin");
		MF>>csc_model.features.sbin;
		// dim
		MF>>Title;
		yuCheck(Title,"dim");
		MF>>csc_model.features.dim;
		// truncation_dim
		MF>>Title;
		yuCheck(Title,"truncation_dim");
		MF>>csc_model.features.truncation_dim;
		// extra_octave
		MF>>Title;
		yuCheck(Title,"extra_octave");
		MF>>tmp;
		csc_model.features.extra_octave = (tmp!=0);
		// bias
		MF>>Title;
		yuCheck(Title,"bias");
		MF>>csc_model.features.bias;
	}

	// bboxpred
	MF>>Title;
	yuCheck(Title,"bboxpred");
	MF>>Sz;
	csc_model.bboxpred.resize(Sz);
	for( int i=0; i<Sz; i++ ){
		MF>>tmp;
		yuCheck(tmp==i+1);
		// x1
		MF>>Title;
		yuCheck(Title,"x1");
		MF>>tmp;
		csc_model.bboxpred[i].x1.resize(tmp);
		for( int j=0; j<tmp; j++ )
			MF>>csc_model.bboxpred[i].x1[j];
		// y1
		MF>>Title;
		yuCheck(Title,"y1");
		MF>>tmp;
		csc_model.bboxpred[i].y1.resize(tmp);
		for( int j=0; j<tmp; j++ )
			MF>>csc_model.bboxpred[i].y1[j];
		// x2
		MF>>Title;
		yuCheck(Title,"x2");
		MF>>tmp;
		csc_model.bboxpred[i].x2.resize(tmp);
		for( int j=0; j<tmp; j++ )
			MF>>csc_model.bboxpred[i].x2[j];
		// y2
		MF>>Title;
		yuCheck(Title,"y2");
		MF>>tmp;
		csc_model.bboxpred[i].y2.resize(tmp);
		for( int j=0; j<tmp; j++ )
			MF>>csc_model.bboxpred[i].y2[j];
	}

	// rootfilters
	MF>>Title;
	yuCheck(Title,"rootfilters");
	MF>>Sz0;
	csc_model.rootfilters.resize(Sz0);
	for( int i=0; i<Sz0; i++ ){
		MF>>tmp;
		yuCheck(tmp==i+1);
		// blocklabel
		MF>>Title;
		yuCheck(Title,"blocklabel");
		MF>>tmp;
		csc_model.rootfilters[i].blocklabel = tmp;
		// size
		MF>>Title;
		yuCheck(Title,"size");
		MF>>M>>N;
		csc_model.rootfilters[i].size[0] = M;
		csc_model.rootfilters[i].size[1] = N;
		// w
		MF>>Title;
		yuCheck(Title,"w");
		MF>>M>>N>>D;
		csc_model.rootfilters[i].w.create( M, N, CV_32FC(D) );
		pt = (float*)( csc_model.rootfilters[i].w.ptr<float>(0,0) );
		Sz = M * N * D;
		tmp = 0;
		while( tmp<Sz )
			MF>>pt[tmp++];
		// wpca
		MF>>Title;
		yuCheck(Title,"wpca");
		MF>>M>>N>>D;
		csc_model.rootfilters[i].wpca.create( M, N, CV_32FC(D) );
		pt =  (float*)( csc_model.rootfilters[i].wpca.ptr<float>(0,0) );
		Sz = M * N * D;
		tmp = 0;
		while( tmp<Sz )
			MF>>pt[tmp++];
	}

	// offsets
	MF>>Title;
	yuCheck(Title,"offsets");
	MF>>Sz;
	csc_model.offsets.resize(Sz);
	for( int i=0; i<Sz; i++ ){
		MF>>tmp;
		yuCheck(tmp==i+1);
		// w
		MF>>Title;
		yuCheck(Title,"w");
		MF>>csc_model.offsets[i].w;
		// blocklabel
		MF>>Title;
		yuCheck(Title,"blocklabel");
		MF>>csc_model.offsets[i].blocklabel;
	}

	// loc_w
	MF>>Title;
	yuCheck(Title,"loc_w");
	MF>>Sz;
	csc_model.loc_w.resize(Sz);
	{
		// TODO
	}

	// loc
	MF>>Title;
	yuCheck(Title,"loc");
	MF>>Sz;
	csc_model.loc.resize(Sz);
	for( int i=0; i<Sz; i++ ){
		MF>>tmp;
		yuCheck(tmp==i+1);
		// blocklabel
		MF>>Title;
		yuCheck(Title,"blocklabel");
		MF>>csc_model.loc[i].blocklabel;
		// w
		MF>>Title;
		yuCheck(Title,"w");
		MF>>tmp;
		csc_model.loc[i].w.resize(tmp);
		for( int j=0; j<tmp; j++ )
			MF>>csc_model.loc[i].w[j];
	}

	// components
	MF>>Title;
	yuCheck(Title,"components");
	MF>>Sz;
	csc_model.components.resize(Sz);
	for( int i=0; i<Sz; i++ ){
		MF>>tmp;
		yuCheck(tmp==i+1);
		// rootindex
		MF>>Title;
		yuCheck(Title,"rootindex");
		MF>>csc_model.components[i].rootindex;
		// offsetindex
		MF>>Title;
		yuCheck(Title,"offsetindex");
		MF>>csc_model.components[i].offsetindex;
		// parts
		MF>>Title;
		yuCheck(Title,"parts");
		MF>>tmp;
		csc_model.components[i].parts.resize(tmp);
		for( int j=0; j<tmp; j++ ){
			MF>>M;
			yuCheck(M==j+1);
			// partindex
			MF>>Title;
			yuCheck(Title,"partindex");
			MF>>csc_model.components[i].parts[j].partindex;
			// defindex
			MF>>Title;
			yuCheck(Title,"defindex");
			MF>>csc_model.components[i].parts[j].defindex;
			
			// -1
			csc_model.components[i].parts[j].defindex--;
			csc_model.components[i].parts[j].partindex--;
		}

		// -1
		csc_model.components[i].rootindex--;
		csc_model.components[i].offsetindex--;
	}

	// partfilters
	MF>>Title;
	yuCheck(Title,"partfilters");
	MF>>Sz0;
	csc_model.partfilters.resize(Sz0);
	for( int i=0; i<Sz0; i++ ){
		MF>>tmp;
		yuCheck(tmp==i+1);
		// blocklabel
		MF>>Title;
		yuCheck(Title,"blocklabel");
		MF>>csc_model.partfilters[i].blocklabel;
		// w
		MF>>Title;
		yuCheck(Title,"w");
		MF>>M>>N>>D;
		csc_model.partfilters[i].w.create( M, N, CV_32FC(D) );
		pt = (float*)( csc_model.partfilters[i].w.ptr<float>(0,0) );
		Sz = M * N * D;
		tmp = 0;
		while( tmp<Sz )
			MF>>pt[tmp++];
		// wpca
		MF>>Title;
		yuCheck(Title,"wpca");
		MF>>M>>N>>D;
		csc_model.partfilters[i].wpca.create( M, N, CV_32FC(D) );
		pt = (float*)( csc_model.partfilters[i].wpca.ptr<float>(0,0) );
		Sz = M * N * D;
		tmp = 0;
		while( tmp<Sz )
			MF>>pt[tmp++];
	}

	// defs
	MF>>Title;
	yuCheck(Title,"defs");
	MF>>Sz;
	csc_model.defs.resize(Sz);
	for( int i=0; i<Sz; i++ ){
		MF>>tmp;
		yuCheck(tmp==i+1);
		// blocklabel
		MF>>Title;
		yuCheck(Title,"blocklabel");
		MF>>csc_model.defs[i].blocklabel;
		// anchor
		MF>>Title;
		yuCheck(Title,"anchor");
		MF>>csc_model.defs[i].anchor[0]>>csc_model.defs[i].anchor[1];
		// w
		MF>>Title;
		yuCheck(Title,"w");
		MF>>tmp;
		yuCheck(tmp==4);
		for( int j=0; j<4; j++ )
			MF>>csc_model.defs[i].w[j];
	}

	// cascade
	MF>>Title;
	yuCheck(Title,"cascade");
	{
		// thresh
		MF>>Title;
		yuCheck(Title,"thresh");
		MF>>csc_model.cascade.thresh;
		// order
		MF>>Title;
		yuCheck(Title,"order");
		MF>>tmp;
		csc_model.cascade.order.resize(tmp);
		for( int i=0; i<tmp; i++ ){
			MF>>M; 
			yuCheck(M==i+1);
			MF>>M;
			csc_model.cascade.order[i].resize(M);
			for( int j=0; j<M; j++ )
				MF>>csc_model.cascade.order[i][j];
		}
		// t
		MF>>Title;
		yuCheck(Title,"t");
		MF>>tmp;
		csc_model.cascade.t.resize(tmp);
		for( int i=0; i<tmp; i++ ){
			MF>>M;
			yuCheck(M==i+1);
			MF>>M;
			csc_model.cascade.t[i].resize(M);
			for( int j=0; j<M; j++ )
				MF>>csc_model.cascade.t[i][j];
		}

		// -1
		for( int i=0; i<csc_model.cascade.order.size(); i++ )
			for( int j=0; j<csc_model.cascade.order[i].size(); j++ )
				csc_model.cascade.order[i][j] --;
	}
	
	// END
	MF>>Title;
	yuCheck(Title,"END");

	//ifstream fin("clustercenters.txt");
	ifstream fc2("C2.txt");

	//int M,N;
	fc2>>M>>N;

	//transpose
	csc_model.C2.create(M,N,CV_32FC1);

	float * C2_pt = csc_model.C2.ptr<float>(0,0);

	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
		{
			fc2>>*C2_pt;
			++C2_pt;
		}

	ifstream fc1("C1.txt");

	fc1>>M>>N;

	//transpose
	csc_model.C1.create(M,N,CV_32FC1);

	float * C1_pt = csc_model.C1.ptr<float>(0,0);

	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
		{
			fc1>>*C1_pt;
			++C1_pt;
		}

	
	return;
}