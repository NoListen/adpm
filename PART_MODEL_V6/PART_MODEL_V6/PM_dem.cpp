#include "PM_dem.h"

PM_dem::PM_dem( string model_file )
{
	loadCscModel( model_file, model );

	numrootfilters = model.rootfilters.size();
	rootfilters.resize( numrootfilters );
	for( int i=0; i<numrootfilters; i++ )
		rootfilters[i] = model.rootfilters[i].wpca;

	rootscores.resize( model.numcomponents );

	model.rootlut.resize(model.numcomponents);

	for (int i = 0; i < model.partfilters.size(); ++i)
	{
		model.partfilters[i].wlut = filter2lut(model.partfilters[i].w,model.C2);
	}
	
	for (int i = 0; i < model.rootfilters.size(); ++i)
	{
		model.rootlut[i] = filter2lut(model.rootfilters[i].w,model.C2);
	}

	model.T1.create(511,511,CV_32SC1);
	model.T2.create(511,511,CV_32FC1);
	float uu[9] = {1.0000, 
		0.9397, 
		0.7660, 
		0.500, 
		0.1736, 
		-0.1736, 
		-0.5000, 
		-0.7660, 
		-0.9397};
	float vv[9] = {0.0000, 
		0.3420, 
		0.6428, 
		0.8660, 
		0.9848, 
		0.9848, 
		0.8660, 
		0.6428, 
		0.3420};
	
    for (int i = 0; i < 511; ++i)
        for (int j = 0; j < 511; ++j)
    {
        int dx = i-255;
        int dy = j-255;
        int v = dx*dx + dy*dy;
        float best_dot = 0;;
        int best_o = 0;
        for (int o = 0; o < 9; ++o)
        {
            float dot = uu[o]*dx + vv[o]*dy;
            if (dot > best_dot)
            {
                best_dot = dot;
                best_o = o;
            }
            else if (-dot > best_dot) 
                {
                    best_dot = -dot;
                    best_o = o+9;
                }
        }
		model.T1.at<int>(i,j) = best_o;
		model.T2.at<float>(i,j) = sqrtf(v);
		//cout<<'x'<<endl;
		//cout<<model.T1.at<int>(i,j)<<endl;
    }
	//for debug
	//ofstream fout("cartest.txt");
	//float * test = model.partfilters[2].wlut.ptr<float>(0,0); // 3 dims
	//for (int i = 0; i < 25; ++i)
	//	fout<<*(test+i)<<' ';

	//cout<<model.partfilters[1].wlut<<endl;
	//cout<<"haha"<<endl;
}

Mat	PM_dem::prepareImg( Mat &img_uint8 )
{
	if( img_uint8.depth()!=CV_8U || (img_uint8.channels()!=1 && img_uint8.channels()!=3) ){
		printf("Function only takes as input an image of 1 or 3 channels of uint8 type!");
		throw	runtime_error("");
	}
	if( img_uint8.channels()==1 )
		cvtColor( img_uint8, img_color, CV_GRAY2RGB );
	else if( img_uint8.channels()==3 )
		img_color = img_uint8;
	Mat	img;
	img_color.convertTo( img, CV_32FC3 );
	return	img;
}

void	PM_dem::detect( Mat &img, float score_thresh, bool show_hints, bool show_img, string save_img )
{
	if( score_thresh==DEFAULT_THRESH )
		model.threshing = model.thresh;
	else
		model.threshing = score_thresh;

	hints = show_hints;

	// 1. Feature pyramid <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	prag_start = yuGetCurrentTime('M');
	if( hints ){
		printf("Calculating feature pyramid ...\n");
		start_clock = prag_start;
	}

	featpyramid2( img, model, pyra );	

	if( hints ){
		end_clock = yuGetCurrentTime('M');
		printf("Time for _featpyramid is %gs\n",(end_clock-start_clock)/1000.f);		
	}

	// 2. Compute PCA projection of the feature pyramid <<<<<<<<<<<<<<
	if( hints ){
		printf("Compute PCA projection of the feature pyramid ...\n");
		start_clock = end_clock;
	}
	
	/*Mat Ctest = model.C2(Rect(0,155,32,1));
	cout<<Ctest;*/

	//project_pyramid( model, pyra );

	
	if( hints ){
		end_clock = yuGetCurrentTime('M');
		printf("Time for _project_pyramid() is %gs\n",(end_clock-start_clock)/1000.f);
	}

	if (hints)
	{
		end_clock = start_clock;
		printf("QT\n");
	}

	qtpyra(model,pyra);
	
	if (hints)
	{
		end_clock = yuGetCurrentTime('M');
		printf("%gs\n",(end_clock-start_clock)/1000.f);
	}
	
	if( pyra.num_levels!=pyra.feat.size() ){
		printf("pyra.num_levels!=pyra.feat.size()\n");
		throw	runtime_error("");	
	}


	// 3. Precompute location/scale scores <<<<<<<<<<<<<<<<<<<<<<<
	Mat	loc_f = loc_feat( model, pyra.num_levels );
	pyra.loc_scores.resize( model.numcomponents );
	for( int c=0; c<model.numcomponents; c++ ){
		Mat	loc_w( 1, model.loc[c].w.size(), CV_32FC1, &(model.loc[c].w[0]) ); // loc_w = model.loc[c].w 
		pyra.loc_scores[c] = loc_w * loc_f; 
	}

	// 4. Gather PCA root filters for convolution <<<<<<<<<<<<<<<<<<<
	if( hints ){
		printf("Gathering PCA root filters for convolution ...\n");
		start_clock = end_clock;
	}

	if( rootscores[0].size()!=pyra.num_levels ){
		vector<Mat>	tmp_rootscores(pyra.num_levels);
		rootscores.assign(model.numcomponents,tmp_rootscores);
	}	

//	ofstream f1("pj.txt");

	int	numrootlocs = 0;
	int	s = 0; // will hold the amount of temp storage needed by cascade()
	for( int i=0; i<pyra.num_levels; i++ ){
		s += pyra.feat[i].rows * pyra.feat[i].cols;
		if( i<model.interval )
			continue;
		static vector<Mat>	scores;
		//fconv( pyra.projfeat[i], rootfilters, 0, numrootfilters, scores );
		fconv_root_qt(model, pyra.qtfeat[i], scores);
		for( int c=0; c<model.numcomponents; c++ ){
			int u = model.components[c].rootindex;
			int v = model.components[c].offsetindex;
			float	tmp = model.offsets[v].w + pyra.loc_scores[c].at<float>(i);
			rootscores[c][i] = scores[u] + Scalar(tmp);
			numrootlocs += scores[u].total();
		}
	}
	cout<<numrootlocs<<endl;
	s = s * model.partfilters.size();

	if( hints ){
		end_clock = yuGetCurrentTime('M');
		printf("Time for gathering PCA root filters is %gs\n",(end_clock-start_clock)/1000.f);
	}

	// 5. Cascade detection in action <<<<<<<<<<<<<<<<<<<<<<<
	if( hints ){
		printf("Cascade detection in action ...\n");
		start_clock = end_clock;
	}

	//Mat coords = cascade(model, pyra, rootscores, numrootlocs, s);
	Mat coords = cascade_qt(model, pyra, rootscores, numrootlocs, s);
	//cout<<coords;
	//cout<<"??"<<endl;

	if( hints ){
		end_clock = yuGetCurrentTime('M');
		printf("Time for _cascade() is %gs\n",(end_clock-start_clock)/1000.f);
		if( coords.empty() ){
			printf("No Detection!\n");
			return;
		}
	}

	// 6. Detection results <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	Mat boxes = getboxes( model, img_color, coords );
	Mat x1 = boxes.col(0);
	Mat y1 = boxes.col(1);
	Mat x2 = boxes.col(2);
	Mat y2 = boxes.col(3);
	Mat Score = boxes.col( boxes.cols-1 );	
	detections.resize( x1.rows );
	for( int i=0; i<x1.rows; i++ ){
		detections[i][0] = x1.at<float>(i);
		detections[i][1] = y1.at<float>(i);
		detections[i][2] = x2.at<float>(i);
		detections[i][3] = y2.at<float>(i);
		detections[i][4] = Score.at<float>(i);
	}

	if( hints ){
		prag_end = yuGetCurrentTime('M');
		printf("Total detection time is : %gs\n",(prag_end-prag_start)/1000.f);
	}

	// 6. Draw and show <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<	
	if( show_img || !save_img.empty() ){
		//showboxes( img_color, boxes );

		//
		const int fontFace = CV_FONT_HERSHEY_PLAIN;
		const double fontScale = 1;		
		const Scalar drawColor = CV_RGB(255,0,0);
		const Scalar fontColor = CV_RGB(30,250,150);
		//
		for( int i=0; i!=detections.size(); i++ ){
			float		x1 = detections[i][0], y1 = detections[i][1], x2 = detections[i][2], y2 = detections[i][3];
			float		_score = detections[i][4];			
			//
			Point2f		UL( x1, y1 );
			Point2f		BR( x2, y2 );
			rectangle( img_color, UL, BR, drawColor, 2 );
			printf("----------------------------\n");
			printf("%g  %g  %g  %g  %g\n", x1, y1, x2, y2, _score );
			//
			x1 = int(x1*10+0.5) / 10.f; // ½ö±£Áô1Î»Ð¡Êý
			y1 = int(y1*10+0.5) / 10.f;
			x2 = int(x2*10+0.5) / 10.f;
			y2 = int(y2*10+0.5) / 10.f;
			_score = int(_score*100+0.5) / 100.f;			
			//
			char	buf[50] = { 0 };
			sprintf_s( buf, 50, "%d", i );
			string   text = buf;
			int		  baseline = 0;
			Size	  textSize = getTextSize( text, fontFace, fontScale, 1, &baseline );
			Point2f   textOrg2( x1, y1+textSize.height+2 );
			putText( img_color, text, textOrg2, fontFace, fontScale, fontColor );
			//				
			sprintf_s( buf, 50, "%d %g %g %g %g %g", i, x1, y1, x2, y2, _score );
			text = buf;			
			textSize = getTextSize( text, fontFace, fontScale, 1, &baseline );
			Point2f	  textOrg(5,(i+1)*(textSize.height+3));
			putText( img_color, text, textOrg, fontFace, fontScale,	fontColor );
		}
		{
			char	buf[30] = { 0 };
			sprintf_s( buf, 30, "time : %gs", (prag_end-prag_start)/1000.f );
			string		time_text = buf;
			int			baseline = 0;
			Size		textSize = getTextSize( time_text, fontFace, fontScale, 1, &baseline );
			Point2f	time_orig(5,(detections.size()+1)*(textSize.height+3));
			putText( img_color, time_text, time_orig, fontFace, fontScale, fontColor );
		}
		if( show_img )
			imshow( "OK", img_color );		
		if( !save_img.empty() )
			imwrite( save_img, img_color );

	}
	
}