#include "../PartModel_funs.h"

void	project_pyramid( const CSC_MODEL &model, FEATURE_PYRAMID &pyra )
//pyra = project_pyramid(model, pyra)
//
//Project feature pyramid pyra onto PCA eigenvectors stored
//in model.coeff.
{
	pyra.projfeat.resize( pyra.num_levels );
	for( int i=0; i<pyra.num_levels; i++ )
		pyra.projfeat[i] = project( pyra.feat[i], model.pca_coeff );
}
