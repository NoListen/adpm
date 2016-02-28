//#include "../PM_type.h"
#include "../PartModel_funs.h"

#include <cstdio>
#include <string.h>

/*
 * Fast image subsampling.
 * This is used to construct the feature pyramid.
 */
//namespace PM_type{

// struct used for caching interpolation values
struct alphainfo {
  int si, di;
  float alpha;
};

// copy src into dst using pre-computed interpolation values
void alphacopy(float *src, float *dst, struct alphainfo *ofs, int n, int step, int x, int c) {
  //struct alphainfo *end = ofs + n;
  int k = n;
  while (k) {
	x = x;
	c = c;
    dst[ofs->di] += ofs->alpha * src[(ofs->si)*step];
    --k;
	++ofs;
  }
}

// resize along each column
// result is transposed, so we can apply it twice for a complete resize
void resize1dtran(float *src, int sheight, float *dst, int dheight, 
		  int width, int chan) {
  float scale = (float)dheight/(float)sheight;
  float invscale = (float)sheight/(float)dheight;
  int step = width*chan;
  // we cache the interpolation values since they can be 
  // shared among different columns
  int len = (int)ceil(dheight*invscale) + 2*dheight;
  alphainfo *ofs = new alphainfo[len];
  int k = 0;
  for (int dy = 0; dy < dheight; dy++) {
    float fsy1 = dy * invscale;
    float fsy2 = fsy1 + invscale;
    int sy1 = (int)ceil(fsy1);
    int sy2 = (int)fsy2;       

    if (sy1 - fsy1 > 1e-3) {
      assert(k < len);
      assert(sy1 -1 >= 0);
      //ofs[k].di = dy*width;
	  ofs[k].di = dy*chan;
      ofs[k].si = sy1-1;
      ofs[k++].alpha = (sy1 - fsy1) * scale;
    }

    for (int sy = sy1; sy < sy2; sy++) {
      assert(k < len);
      assert(sy < sheight);
      //ofs[k].di = dy*width;
	  ofs[k].di = dy*chan;
      ofs[k].si = sy;
      ofs[k++].alpha = scale;
    }

    if (fsy2 - sy2 > 1e-3) {
      assert(k < len);
      assert(sy2 < sheight);
      //ofs[k].di = dy*width;
	  ofs[k].di = dy*chan;
      ofs[k].si = sy2;
      ofs[k++].alpha = (fsy2 - sy2) * scale;
    }
  }

  // resize each column of each color channel
  memset(dst, 0 ,chan*width*dheight*sizeof(float));
  for (int c = 0; c < chan; c++) {
    for (int x = 0; x < width; x++) {
      //float *s = src + c*width*sheight + x*sheight;
	  float *s = src + chan*x + c; 
	  //already the x-th col
	  // each chan*width step a increment
      //float *d = dst + c*width*dheight + x;
	  float *d = dst + chan*dheight*x + c;
	  //need to be in the x-th row
	  //then x-th row + dy*width would be changed into x-th row + chan*di
	  //then si represent the height y
	  // si need to be changed into si*width*chan
      alphacopy(s, d, ofs, k, step, x, c);
    }
  }
}

// main function
// takes a float color image and a scaling factor
// returns resized image
Mat resize(Mat const &src_mat, float const &scale) {
  const int sdims[3] = {src_mat.rows, src_mat.cols, src_mat.channels()};
  const int row_step = src_mat.cols * src_mat.channels();

  float* srcpt = const_cast<float*>(src_mat.ptr<float>(0,0));
  if (src_mat.type() != CV_32FC3)
	  throw runtime_error("Invalid input");

  if (scale > 1)
    throw runtime_error("Invalid scaling factor");  

  int ddims[3];
  ddims[0] = (int)(sdims[0]*scale + 0.5);
  ddims[1] = (int)(sdims[1]*scale + 0.5);
  ddims[2] = sdims[2];
  Mat dst_mat(ddims[0],ddims[1],CV_32FC(ddims[2]));
  float * dst = const_cast<float*>(dst_mat.ptr<float>(0,0));

  float *tmp = new float[ddims[0]*sdims[1]*sdims[2]];
  resize1dtran(srcpt, sdims[0], tmp, ddims[0], sdims[1], sdims[2]);
  resize1dtran(tmp, sdims[1], dst, ddims[1], ddims[0], sdims[2]);
  
  delete tmp;
  const float* an = dst_mat.ptr<float>(0,0);
  return dst_mat;
}

//}

// matlab entry point
// dst = resize(src, scale)
// image should be color with float values
/*void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  if (nrhs != 2)
    mexErrMsgTxt("Wrong number of inputs"); 
  if (nlhs != 1)
    mexErrMsgTxt("Wrong number of outputs");
  plhs[0] = resize(prhs[0], prhs[1]);
}*/



