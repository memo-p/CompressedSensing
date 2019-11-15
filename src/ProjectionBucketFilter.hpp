/*
 * Copyright (C) 2020 Guillaume Perez
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <iostream>
#include <cstdio>
#include <armadillo>
#include <cfloat>    // for DBL_MAX
#include "DTB.hpp"

using namespace arma;
using namespace std;

// /* Algorithm w-bucket^f in the paper */
void project_bucket_filter(double* y, double* w,  double* x,
const unsigned int length, const double a){   

    union DtB r1;
    int * per = (int*)malloc(2*length*sizeof(int));
    int * per2 = per + length;
    int *  perswap;
    int * ptrToFree = per;
    int illength;
    double  tau;
    int currentLength;
    int t[257];
    double s[257];
    double wbs[257];
    double minS[257];
    double maxS[257];  
    int * tmp;        
    tmp = &t[0];
    tmp ++;
    int bucketSize;
    int start;
    
    double sumWY = 0;
    double Ws = 0;
    double lsumWY = 0;
    double lWs = 0;

    int i; 
    int j;
    int over = 0;

    illength = length;       
    int depth = 7;

    for (i = 0; i < 257; ++i){
        t[i] = 0;
        s[i] = 0.;
        wbs[i] = 0.;
        minS[i] = DBL_MAX;
        maxS[i] = DBL_MIN;
    }

    per[0] = 0;
    sumWY = w[0] * y[0];
    Ws = w[0] * w[0];
    r1.val = y[0] / w[0];
    double p = (sumWY - a) / Ws;
    int idx = r1.byte[depth];
    tmp[idx]++;
    wbs[idx] += w[0] * w[0];
    s[idx] += y[0] * w[0];
    minS[idx] = r1.val;
    maxS[idx] = r1.val;
    double wy,ww;

    for (j=i=1; j < length; ++i, ++j){
        if(y[j] > w[j] * p){
            per[i] = j;
            r1.val = y[j] / w[j];
            wy = w[j] * y[j];
            ww = w[j] * w[j];
            sumWY += wy;
            Ws += ww;
            p = (sumWY - a) / Ws;
            if(p <= ((wy - a) / (ww))){
                sumWY =  wy;
                Ws = ww;
                p = (sumWY - a) / Ws;
            }
            ++tmp[r1.byte[depth]];
            s[r1.byte[depth]] += wy;
            wbs[r1.byte[depth]] += ww;
            minS[r1.byte[depth]] = (minS[r1.byte[depth]] < r1.val)? minS[r1.byte[depth]] : r1.val;
            maxS[r1.byte[depth]] = (maxS[r1.byte[depth]] > r1.val)? maxS[r1.byte[depth]] : r1.val;
        }else{
            --i;
        }
    }

    tau = - a;
    illength = i;
    sumWY = 0;
    Ws = 0;
    for (depth = 7; depth >= 0; depth --){
        
        for (i = 1; i < 256; ++i){                  // Count sort.
            tmp[i] = tmp[i] + tmp[i-1];
        }
        for (i = 0; i < illength; ++i){      
            r1.val = y[per[i]] / w[per[i]];
            per2[t[r1.byte[depth]]++] = per[i];           
        }

        perswap = per2;               // Swap temporary y/w vector
        per2 = per;
        per = perswap;
        currentLength = illength;

        for (i = 255; i >= 0; --i){ // t[i] is the starting point of the i+1 values (because of the ++ )
            start = (i == 0)? 0 : t[i-1];
            bucketSize = currentLength - start; 
            currentLength -= bucketSize;
            if (bucketSize == 0){               
                continue;
            }
            if (tau > maxS[i]){ // Best possible remaining value is dominatied: end
                over = 1;
                break;
            }
            if ((sumWY + s[i] - a) / (Ws + wbs[i]) < minS[i]){  // try keeping the min of b
                sumWY += s[i];
                Ws += wbs[i];
                tau = (sumWY - a) / Ws;
                continue;
            }
            per += start;
            per2 += start;
            illength = bucketSize;
            break;
        }  
        depth--;
        if (depth < 0 || over == 1 || i < 0)
        {
            break;
        }
        for (i = 0; i < 257; ++i){
            t[i] = 0;
            s[i] = 0.;
            wbs[i] = 0.;
            minS[i] = DBL_MAX;
            maxS[i] = DBL_MIN;
        }
        start = illength-1;
        wy = y[per[start]] * w[per[start]];
        ww = w[per[start]] * w[per[start]];
        r1.val = y[per[start]] / w[per[start]];
        lsumWY = wy;
        lWs =  ww;
        tau = (lsumWY + sumWY - a) / (Ws + lWs);
        ++tmp[r1.byte[depth]];
        wbs[r1.byte[depth]] += ww;
        s[r1.byte[depth]] += wy;
        minS[r1.byte[depth]] = r1.val;
        maxS[r1.byte[depth]] = r1.val;
        for (i = illength-2; i >= 0; --i){
            if (y[per[i]] > w[per[i]] * tau){
                r1.val = y[per[i]] / w[per[i]];
                wy = y[per[i]] * w[per[i]];
                ww = w[per[i]] * w[per[i]];
                lsumWY += wy;
                lWs += ww;
                tau = (lsumWY + sumWY - a) / (Ws + lWs);
                
                s[r1.byte[depth]] += wy;
                wbs[r1.byte[depth]] += ww;
                ++tmp[r1.byte[depth]];
                minS[r1.byte[depth]] = (minS[r1.byte[depth]] < r1.val)? minS[r1.byte[depth]] : r1.val;
                maxS[r1.byte[depth]] = (maxS[r1.byte[depth]] > r1.val)? maxS[r1.byte[depth]] : r1.val;
            }else{
                per[i] = per[--illength];
            }
        }
        depth++;
    }
    // printf("%f\n",tau);

    for (i=0; i<length; i++)
        x[i]=(y[i] > w[i]*tau)? y[i]-w[i]*tau : 0.0; 
    free(ptrToFree);
}