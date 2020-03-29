#include <float.h>
#include <math.h>

#include "DTB.hpp"

namespace proj {

void ProjB(const datatype* y, datatype* x, const unsigned int length,
           const double a) {
  union DtB* r1 = new DtB[length];
  union DtB* r1ptr = r1;
  union DtB* r2 = (union DtB*)x;
  union DtB* auxSwap;
  int plength;
  int illength;
  double p;
  double tau;
  int currentLength;
  int count = 0;
  int t[257];
  double s[257];
  double minS[257];
  double maxS[257];
  union DtB* tmpswap;
  int* tmp;
  tmp = &t[0];
  tmp++;
  int bucketSize;
  int start;

  int i;
  int j;
  int over = 0;

  illength = length;
  int depth = DATASIZE - 1;

  for (i = 0; i < 257; ++i) {
    t[i] = 0;
    s[i] = 0.;
    minS[i] = DBL_MAX;
    maxS[i] = DBL_MIN;
  }

  for (i = 0; i < length; i++) {
    r1[i].val = y[i];
    tmp[r1[i].byte[depth]]++;
    s[r1[i].byte[depth]] += r1[i].val;
    minS[r1[i].byte[depth]] = (minS[r1[i].byte[depth]] < r1[i].val)
                                  ? minS[r1[i].byte[depth]]
                                  : r1[i].val;
    maxS[r1[i].byte[depth]] = (maxS[r1[i].byte[depth]] > r1[i].val)
                                  ? maxS[r1[i].byte[depth]]
                                  : r1[i].val;
  }

  tau = -a;
  illength = length;
  for (depth = DATASIZE - 1; depth >= 0; depth--) {
    for (i = 1; i < 256; ++i) {  // Count sort.
      tmp[i] = tmp[i] + tmp[i - 1];
    }
    for (i = 0; i < illength; ++i) {
      r2[t[r1[i].byte[depth]]++] = r1[i];
    }

    tmpswap = r2;
    r2 = r1;
    r1 = tmpswap;
    currentLength = illength;

    for (i = 255; i >= 0; --i) {  // t[i] is the starting point of the i+1
                                  // values (because of the ++ )
      start = (i == 0) ? 0 : t[i - 1];
      bucketSize = currentLength - start;
      currentLength -= bucketSize;
      if (bucketSize == 0) {
        continue;
      }
      if (tau / count >
          maxS[i]) {  // Best possible remaining value is dominatied: end
        over = 1;
        break;
      }
      if ((tau + s[i]) / (count + bucketSize) <
          minS[i]) {  // try keeping the min of b
        tau += s[i];
        count += bucketSize;
        continue;
      }
      r1 += start;
      r2 += start;
      illength = bucketSize;
      break;
    }
    depth--;
    if (depth < 0 || over == 1 || i < 0) {
      break;
    }
    for (i = 0; i < 257; ++i) {
      t[i] = 0;
      s[i] = 0.;
      minS[i] = DBL_MAX;
      maxS[i] = DBL_MIN;
    }
    for (i = 0; i < illength; ++i) {
      tmp[r1[i].byte[depth]]++;
      s[r1[i].byte[depth]] += r1[i].val;
      minS[r1[i].byte[depth]] = (minS[r1[i].byte[depth]] < r1[i].val)
                                    ? minS[r1[i].byte[depth]]
                                    : r1[i].val;
      maxS[r1[i].byte[depth]] = (maxS[r1[i].byte[depth]] > r1[i].val)
                                    ? maxS[r1[i].byte[depth]]
                                    : r1[i].val;
    }
    depth++;
  }
  tau /= count;
  for (i = 0; i < length; i++) x[i] = (y[i] > tau ? y[i] - tau : 0.0);
  delete [] r1ptr;
}

}  // namespace proj
