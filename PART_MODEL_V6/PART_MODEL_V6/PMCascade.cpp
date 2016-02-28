#include "PartModel_types.h"

// GLOBALs
static const int S = 4;
static vector<float>	PCONV[2];
static vector<float>	DT[2];
static vector<int>		LOFFCONV;
static vector<int>		LOFFDT;
static vector<vector<float> >	DXDEFCACHE;
static vector<vector<float> >	DYDEFCACHE;
static vector<short>		DXAM[2]; // int
static vector<short>		DYAM[2]; // int
//
static float *PCONV_pt[2], *DT_pt[2];
static int *LOFFCONV_pt, *LOFFDT_pt;
static short *DXAM_pt[2], *DYAM_pt[2];
static float **DXDEFCACHE_pt, **DYDEFCACHE_pt;
//
static int padx, pady;
static int numcomponents, numpartfilters, numdefparams, nlevels;
static vector<int>		featdimsprod;
static vector<int>		featdims[2];
static vector<int>		numparts;
static vector<int>		partfilterdims[2];
static vector<vector<float> >	pcascore;
static vector<vector<INTS_2> > anchors;
static vector<vector<int> >		pfind, defind;
static vector<const float*>		loc_scores;
// 
static int *featdimsprod_pt, *featdims_pt[2], *numparts_pt, *partfilterdims_pt[2];
static float** pcascore_pt;
static int **pfind_pt, **defind_pt;
static int *cascade_order;
static float *cascade_t;
static int cascade_order_sz[2], cascade_t_sz[2];
static float *model_offsets;
//
float yuConv( const Mat &A, const Mat &B, const int x, const int y );

static inline float partscore( int L, int defindex, int pfind, int x, int y, int pca, float defthresh, 
				const CSC_MODEL &model, const FEATURE_PYRAMID &pyra );

static inline float pconvdt(int L, int probex, int probey, int filterind, int defindex, int xstart, int xend, int ystart, int yend, int pca, float defthresh, 
			  const CSC_MODEL &model, const FEATURE_PYRAMID &pyra ) ;

//============//============//============
int cnt, cnt2, cnt3;
//============//============//============
//============//============//============
Mat	cascade( const CSC_MODEL &model, const FEATURE_PYRAMID &pyra, 
			const vector<vector<Mat> > &rootscores, const int numrootlocs, const int s )
{
	static bool LOAD_MODEL = true;
	if( LOAD_MODEL ){
		LOAD_MODEL = false;
		numcomponents = model.components.size();
		numpartfilters = model.partfilters.size();
		numdefparams = model.defs.size();
		
		DXDEFCACHE.resize( numdefparams );
		DYDEFCACHE.resize( numdefparams );
		DXDEFCACHE_pt = new float *[numdefparams];
		DYDEFCACHE_pt = new float *[numdefparams];
		for( int i=0; i<numdefparams; i++ ){
			DXDEFCACHE[i].resize( 2*S+1 );
			DYDEFCACHE[i].resize( 2*S+1 );
			DXDEFCACHE_pt[i] = &DXDEFCACHE[i][0];
			DYDEFCACHE_pt[i] = &DYDEFCACHE[i][0];
			const FLOATS_4	&def = model.defs[i].w;
			for( unsigned int j=0; j<DXDEFCACHE[0].size(); j++ ){
				int tmp = j - S;
				int tmp2 = tmp * tmp;
				DXDEFCACHE_pt[i][j] = -def[0]*tmp2 - def[1]*tmp;
				DYDEFCACHE_pt[i][j] = -def[2]*tmp2 - def[3]*tmp;
			}
		}
		
		numparts.resize( numcomponents );
		numparts_pt = &numparts[0];
		for( int i=0; i<numcomponents; i++ )
			numparts_pt[i] = model.components[i].parts.size();
		for( int i=0; i<2; i++ ){
			partfilterdims[i].resize( numpartfilters );
			partfilterdims_pt[i] = &partfilterdims[i][0];
		}
		for( int i=0; i<numpartfilters; i++ ){
			int M = model.partfilters[i].w.rows, N = model.partfilters[i].w.cols;
			partfilterdims_pt[0][i] = M;
			partfilterdims_pt[1][i] = N;
		}
		pcascore.resize( numcomponents );
		pcascore_pt = new float *[numcomponents];
		for( int i=0; i<numcomponents; i++ ){
			pcascore[i].resize( numparts_pt[i]+1 );
			pcascore_pt[i] = &pcascore[i][0];
		}
		anchors.resize( numcomponents );
		defind.resize( numcomponents );		
		pfind.resize( numcomponents );
		defind_pt = new int *[numcomponents];
		pfind_pt = new int *[numcomponents];
		for( int i=0; i<numcomponents; i++ ){
			anchors[i].resize( numparts_pt[i] );
			pfind[i].resize( numparts_pt[i] );
			defind[i].resize( numparts_pt[i] );
			pfind_pt[i] = &pfind[i][0];
			defind_pt[i] = &defind[i][0];
			for( int j=0; j<numparts_pt[i]; j++ ){
				int dind = model.components[i].parts[j].defindex;
				int pind = model.components[i].parts[j].partindex;
				pfind_pt[i][j] = pind;
				defind_pt[i][j] = dind;
				anchors[i][j] = model.defs[dind].anchor;
			}
		}

		cascade_order_sz[0] = model.cascade.order.size();
		cascade_order_sz[1] = model.cascade.order[0].size();
		cascade_order = new int [cascade_order_sz[0]*cascade_order_sz[1]];		
		int k=0;
		for( int i=0; i<cascade_order_sz[0]; i++ )
			for( int j=0; j<cascade_order_sz[1]; j++ )
				cascade_order[k++] = model.cascade.order[i][j];
		//??? what happened
		cascade_t_sz[0] = model.cascade.t.size();
		cascade_t_sz[1] = model.cascade.t[0].size();
		cascade_t = new float [cascade_t_sz[0]*cascade_t_sz[1]];
		k = 0;
		for( int i=0; i<cascade_t_sz[0]; i++ )
			for( int j=0; j<cascade_t_sz[1]; j++ )
				cascade_t[k++] = model.cascade.t[i][j];
		model_offsets = new float [model.offsets.size()];
		for( unsigned int i=0; i<model.offsets.size(); i++ )
			model_offsets[i] = model.offsets[i].w;
	}
	nlevels = pyra.num_levels - model.interval;
	//=================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	static bool UPDATE = true;
	static int sss;	
	if( !UPDATE ){
		int padx1 = pyra.padx;
		int pady1 = pyra.pady;
		int nlevels1 = pyra.num_levels - model.interval;
		UPDATE = ( padx1!=padx || pady1!=pady || nlevels1!=nlevels || s!=sss ); 
	}	
	if( UPDATE ){
		UPDATE = false;
		sss = s;
		// PART 1. initialize global variables
		padx = pyra.padx;
		pady = pyra.pady;
		for( int i=0; i<2; i++ ){
			PCONV[i].assign(s,-FLOAT_INF); 
			DT[i].assign(s,-FLOAT_INF); 
			featdims[i].resize( pyra.num_levels );
			PCONV_pt[i] = &PCONV[i][0];
			DT_pt[i] = &DT[i][0];
			featdims_pt[i] = &featdims[i][0];
		}

		LOFFCONV.resize( pyra.num_levels+1 );
		LOFFDT.resize( pyra.num_levels+1 );
		featdimsprod.resize( pyra.num_levels );
		LOFFCONV_pt = &LOFFCONV[0];
		LOFFDT_pt = &LOFFDT[0];
		featdimsprod_pt = &featdimsprod[0];

		LOFFCONV_pt[0] = 0;
		LOFFDT_pt[0] = 0;
		for(unsigned  int i=1; i<LOFFCONV.size(); i++ ){
			int M = pyra.feat[i-1].rows, N = pyra.feat[i-1].cols;
			featdims_pt[0][i-1] = M;
			featdims_pt[1][i-1] = N;
			featdimsprod_pt[i-1] = M*N;
			LOFFCONV_pt[i] = LOFFCONV_pt[i-1] + numpartfilters*featdimsprod_pt[i-1];
			LOFFDT_pt[i] = LOFFDT_pt[i-1] + numdefparams*featdimsprod_pt[i-1];
		}

		for( int i=0; i<2; i++ ){
			int		LOF_END = LOFFDT_pt[LOFFDT.size()-1];
			DXAM[i].resize( LOF_END );
			DYAM[i].resize( LOF_END );
			DXAM_pt[i] = &DXAM[i][0];
			DYAM_pt[i] = &DYAM[i][0];
		}	

		loc_scores.resize( pyra.loc_scores.size() );
		for( unsigned int i=0; i<pyra.loc_scores.size(); i++ ){
			if( !pyra.loc_scores[i].isContinuous() ){
				printf("WRONG IN PMCascade() : pyra.loc_scores not continuous!\n");
				throw runtime_error("");
			}
			loc_scores[i] = pyra.loc_scores[i].ptr<float>(0,0);
		}	
	}
	else{
		Scalar	initval = Scalar(-FLOAT_INF);
		Mat		tmp( 1, s, CV_32FC1, (void*)PCONV_pt[0] );
		tmp = initval;
		tmp.data = (uchar*)( DT_pt[0] );
		tmp = initval;
		tmp.data = (uchar*)( PCONV_pt[1] );
		tmp = initval;
		tmp.data = (uchar*)( DT_pt[1] );
		tmp = initval;
	}
	//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<=================================

	// OUTPUT
	vector<float>	coords;
	cnt = 0, cnt2 = 0, cnt3 = 0;

	// PART 2. process each model component and pyramid level	
	for( int comp=0; comp<numcomponents; comp++ ){
		int numstages = 2*numparts_pt[comp] + 2;
		for( int plevel=0; plevel<nlevels; plevel++ ){
			// 1. root filter pyramid level
			int rlevel = plevel + model.interval;
			float loc_score = loc_scores[comp][rlevel];
			float offsets = model_offsets[comp];
			float bias = offsets + loc_score;
			// 2. get pointer to the scores of the first PCA filter
			const Mat &rootscore = rootscores[comp][rlevel];
			int M = rootscore.rows, N = rootscore.cols;
			int rx0 = int( 0.9999f + padx/2.f );
			int ry0 = int( 0.9999f + pady/2.f );			
			if( !rootscore.isContinuous() ){
				printf("WRONG IN PMCascade() : pyra.loc_scores not continuous!\n");
				throw runtime_error("");
			}
			const float	*rtscore = rootscore.ptr<float>(0,0);
			const int rtstep = rootscore.step1();
			// 3. process each location in the current pyramid level
			for( int rx=rx0; rx<N-rx0; rx++ ){
				for( int ry=ry0; ry<M-ry0; ry++ ){
					//printf("comp = %d, plevel = %d, rx = %d, ry = %d\n",comp,plevel,rx,ry);
					// a. get stage 1 score (PCA root + component offset)
					//float score = rootscore.at<float>(ry,rx);
					float score = rtscore[ry*rtstep+rx];
					// b. record score of PCA filter 
					pcascore_pt[comp][0] = score - bias;
					// c. cascade stages 2 through 2*numparts_pt+2					
					int stage=0;
					while( ++stage<numstages ){
					//for( stage=1; stage<numstages; stage++ ){
						//printf("stage = %d\n",stage);
						// d. check for hypothesis pruning
						if( score<cascade_t[comp*cascade_t_sz[1]+2*stage-1] ) 
							break;
						// e. pca=1 if we're placing pca filters. pca=0 if we're placing "full"/non-pca filters
						int pca = (stage<numparts_pt[comp]+1 ? 1 : 0);
						// f. get the part# used in this stage. root parts have index -1, non-root parts are indexed 0:numparts
						int part = cascade_order[comp*cascade_order_sz[1]+stage];
						if( part<0 ){
							// g. we just finished placing all PCA filters
							const Mat &A = pyra.feat[rlevel];
							//×÷ÕßÄã²»Êì°¡£¡£¡£¡
							const Mat &B = model.rootfilters[comp].w;
							float rscore = yuConv(A,B,rx,ry); cnt3++;
							score += rscore - pcascore_pt[comp][0];
						}
						else{
							// h. place a non-root filter (either PCA or non-PCA)
							int px = 2*rx + anchors[comp][part][0];
							int py = 2*ry + anchors[comp][part][1];
							// i. lookup the filter and deformation model used by this part
							int		filterind = pfind_pt[comp][part];
							int		dind = defind_pt[comp][part];
							float	defthresh = cascade_t[comp*cascade_t_sz[1]+2*stage] - score; 
							// j. ps = partscore(plevel, defind, filterind, px, py, pca, defthresh); 
							//printf("ps_before\n");
							float	ps = partscore(plevel, dind, filterind, px, py, pca, defthresh,model,pyra); 
							//printf("ps_after\n");
							if( pca==1 ){
								// record PCA filter score and update hypothesis score with ps
								pcascore_pt[comp][part+1] = ps;
								score += ps;
							}
							else
								// update hypothesis score by replacing the PCA filter score with ps
								score += ps - pcascore_pt[comp][part+1];
						}
					}
					// l. check if the hypothesis passed all stages with a final score over the global threshold 
					if( stage==numstages && score>=model.threshing ){
						// m. compute and record image coordinates of the detection window
						float scale = model.sbin / pyra.scales[rlevel];
						float x1 = (rx-padx)*scale;
						float y1 = (ry-pady)*scale;
						int m = model.rootfilters[comp].w.rows;
						int n = model.rootfilters[comp].w.cols;
						float x2 = x1 + n*scale - 1;
						float y2 = y1 + m*scale - 1;
						coords.push_back( x1 );
						coords.push_back( y1 );
						coords.push_back( x2 );
						coords.push_back( y2 );
						// n. compute and record image coordinates of the part filters
						scale = model.sbin / pyra.scales[plevel];
						for( int P=0; P<numparts_pt[comp]; P++ ){
							int probex = 2*rx + anchors[comp][P][0];
							int probey = 2*ry + anchors[comp][P][1];
							int dind = defind_pt[comp][P];
							int offset = LOFFDT_pt[plevel] + dind*featdimsprod_pt[plevel] + (probex-padx)*featdims_pt[0][plevel] + probey-pady;
							//really?
							int px = DXAM_pt[0][offset] + padx;
							int py = DYAM_pt[0][offset] + pady;
							float x1 = (px-2*padx)*scale;
							float y1 = (py-2*pady)*scale;
							float x2 = x1 + partfilterdims_pt[1][P]*scale - 1;
							float y2 = y1 + partfilterdims_pt[0][P]*scale - 1;
							coords.push_back(x1);
							coords.push_back(y1);
							coords.push_back(x2);
							coords.push_back(y2);
						}
						coords.push_back( (float)comp );
						coords.push_back( score );
					}
				} // for( int ry=ry0; ry<M-ry0
			} // for( int rx=rx0; rx<N-rx0; rx++ )
		} // for( int plevel=0; plevel<nlevels
	} // for( int comp=0; comp<numcomponents

	printf("cnt = %d, cnt2 = %d, cnt3 = %d\n",cnt,cnt2,cnt3);
	int len = 4 + 4*numparts_pt[0] + 2;
	if( coords.empty() )
		return	Mat();
	Mat coords2( coords.size()/len, len, CV_32FC1, &coords[0] );
	Mat dets( coords2.rows, coords2.cols, coords2.type() );
	coords2.copyTo( dets );
	return	dets;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
float	SSE_DOT( const Mat &A, const Mat &B );
float yuConv( const Mat &A, const Mat &B, const int x, const int y )
{
	const Mat &Asub = A( Rect(x,y,B.cols,B.rows) );
	float val = 0;
	if( Asub.channels()==32 && B.channels()==32 )
		val = SSE_DOT( Asub, B );
	else
		val = (float)( B.dot( Asub ) );
	return	val;
}

static inline float partscore( int L, int defindex, int pfind, int x, int y, int pca, float defthresh, 
				const CSC_MODEL &model, const FEATURE_PYRAMID &pyra )
{
	// remove virtual padding
	x -= padx;
	y -= pady;
	// check if already computed...
	int offset = defindex*featdimsprod_pt[L] + x*featdims_pt[0][L] + y;
	offset = offset + LOFFDT_pt[L];
	float ptr = DT_pt[pca][offset];
	if( ptr>-FLOAT_INF )
		return	ptr;
	// 3. ...nope, define the bounds of the convolution and distance transform region
	int xstart = x-S, xend = x+S;
	int ystart = y-S, yend = y+S;
	xstart = MAX(xstart,0);
	ystart = MAX(ystart,0);
	int A_dims[2] = { featdims_pt[0][L], featdims_pt[1][L] };
	int B_dims[2] = { partfilterdims_pt[0][pfind], partfilterdims_pt[1][pfind] };
	xend = MIN( xend, A_dims[1]-B_dims[1] );
	yend = MIN( yend, A_dims[0]-B_dims[0] );
	float	val = pconvdt(L, x, y, pfind, defindex, xstart, xend, ystart, yend, pca, defthresh, model, pyra);;
	return	val;
}

static inline float pconvdt(int L, int probex, int probey, int filterind, int defindex, int xstart, int xend, int ystart, int yend, int pca, float defthresh, 
			  const CSC_MODEL &model, const FEATURE_PYRAMID &pyra ) 
{
	// compute convolution of a filter and distance transform of over the resulting values using memorized convolutions and deformation pruning
	const Mat &A = ( pca==0 ? pyra.feat[L] : pyra.projfeat[L] );
	const Mat &B = ( pca==0 ? model.partfilters[filterind].w : model.partfilters[filterind].wpca );
	if( A.channels()!=B.channels() ){
		//printf("Wrong in pconvdt()!");
		throw runtime_error("");
	}

	int ptr_offset0 = LOFFCONV_pt[L] + filterind*featdimsprod_pt[L];

	float mmax = -FLOAT_INF;
	float val;
	int xp = probex, yp = probey, argmax_x = 0, argmax_y = 0;
	while (true)
	{
		int ptr_offset = ptr_offset0 + xp*featdims_pt[0][L] + yp;
		float *ptr = PCONV_pt[pca] + ptr_offset;
		float *DXpt = DXDEFCACHE_pt[defindex] + probex - xp + S;
		float *DYpt = DYDEFCACHE_pt[defindex] + probey - yp + S;
		//if (!argmax_x && !argmax_y)
		//{
			for (int i = -1; i <= 1; ++i)
				for (int j = -1; j <= 1; ++j)
				{
					int nxp = xp+i;
					int nyp = yp+j;
					if (nxp > xend || nxp < xstart || nyp > yend || nyp < ystart)
						continue;
					float defcost = *(DXpt-i) + *(DYpt-j);
					float *ptr_p = ptr + i*featdims_pt[0][L] + j;
					if (*ptr_p>-FLOAT_INF)
						val = defcost + *ptr_p;
					else if (defcost < defthresh)
						val = -FLOAT_INF;
					else {
						*ptr_p = yuConv(A,B,xp+i,yp+j);
						val = defcost + *ptr_p;
					}

					if (val > mmax)
					{
						mmax = val;
						argmax_x = i;
						argmax_y = j;
					}
				}
		//}
		//else
		//{
		//	int ty = argmax_y;
		//	if (argmax_x)
		//	{
		//		int nxp = xp+argmax_x;
		//		float *ptr_p = ptr + argmax_x*featdims_pt[0][L];
		//		if (nxp <= xend && nxp >= xstart)
		//			for (int j = -1; j <= 1; ++j)
		//			{
		//				int nyp = yp+j;
		//				if (nyp > yend || nyp < ystart)
		//					continue;
		//				float defcost = *(DXpt-argmax_x) + *(DYpt-j);
		//				float *ptr_p = ptr_p + j;
		//				if (*ptr_p>-FLOAT_INF)
		//					val = defcost + *ptr_p;
		//				else if (defcost < defthresh)
		//						val = -FLOAT_INF;
		//				else {
		//					*ptr_p = yuConv(A,B,xp+argmax_x,yp+j);
		//					val = defcost + *ptr_p;
		//				}

		//				if (val > mmax)
		//				{
		//					mmax = val;
		//					argmax_y = j;
		//				}
		//			}
		//	}

		//	if (ty)
		//	{
		//		
		//		int nyp = yp+ty;
		//		if (nyp <= yend && nyp >= ystart)
		//			for (int i = -1; i <= 1; ++i)
		//			{
		//				int nxp = xp+i;		
		//				if (nxp > xend || nxp < xstart)
		//					continue;
		//				float defcost = *(DXpt-i) + *(DYpt-ty);
		//				float *ptr_p = ptr + i*featdims_pt[0][L] + ty;
		//				if (*ptr_p>-FLOAT_INF)
		//					val = defcost + *ptr_p;
		//				else if (defcost < defthresh)
		//						val = -FLOAT_INF;
		//				else {
		//					*ptr_p = yuConv(A,B,xp+i,yp+ty);
		//					val = defcost + *ptr_p;
		//				}

		//				if (val > mmax)
		//				{
		//					mmax = val;
		//					argmax_x = i;
		//					argmax_y = ty;
		//				}
		//			}	
		//	}
		//}
		if (!argmax_x && !argmax_y)
			break;
		xp = xp + argmax_x;
		yp = yp + argmax_y;
		argmax_x = 0;
		argmax_y = 0;
	}

	// record max and argmax for DT
	int offset = LOFFDT_pt[L] + defindex*featdimsprod_pt[L] + probex*featdims_pt[0][L] + probey;
	DXAM_pt[pca][offset] = xp;
	DYAM_pt[pca][offset] = yp;
	DT_pt[pca][offset] = mmax; cnt2++;
	return	mmax;
}

//yuConv( const Mat &A, const Mat &B, const int x, const int y )
//float conv(int x, int y, const Mat &A, const Mat &B)
//{
//	float val = 0;
//	const float * A_pt = A.ptr<float>(0,0		