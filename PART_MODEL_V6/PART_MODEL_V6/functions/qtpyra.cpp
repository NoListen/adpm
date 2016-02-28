#include "../PartModel_funs.h"

void qtpyra( CSC_MODEL &model, FEATURE_PYRAMID &pyra )
{
	vector<Mat> qpyra;

	pyra.qtfeat.resize(pyra.num_levels);
	int padx = pyra.padx;
	int pady = pyra.pady;

	for (int i = 0; i < pyra.num_levels; ++i)
	{
		pyra.qtfeat[i] = qt(pyra.feat[i],model.C1,model.C2,padx,pady);
	}
}