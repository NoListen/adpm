#ifndef PART_MODEL_DEMONSTRATOR_H
#define PART_MODEL_DEMONSTRATOR_H

#include "PartModel_funs.h"

class PM_dem
{
public:
	PM_dem( string model_file );

	Mat	prepareImg( Mat &img_uint8 ); // returns a float type image and set "img_color" with appropriate value

	enum	{ DEFAULT_THRESH = -INT_INF, DEFAULT_MAX_NUM = INT_INF };

	void	detect( Mat &img, float score_thresh = DEFAULT_THRESH, bool show_hints = true,  bool show_img = true, string save_img = "" );

	vector<FLOATS_5>	detections;

	/* All useful stuff have been declared above. */
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	/* Below are not supposed to be used. */

public:
	CSC_MODEL		model;	
	FEATURE_PYRAMID		pyra;

	Mat		img_color;

	bool		hints;
	int			start_clock;
	int			end_clock;
	int			prag_start;
	int			prag_end;

	int			numrootfilters;
	vector<Mat>	rootfilters;
	vector<vector<Mat> >	rootscores;	
};

#endif