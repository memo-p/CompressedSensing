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
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>
#include <armadillo>
#include "ProjectionSort.hpp"
#include "ProjectionBucket.hpp"
#include "ProjectionBucketFilter.hpp"
#include "ProjectionWSplit.hpp"

using namespace std;
using namespace arma;

namespace proj {


void project(vec& y, vec& w,  vec& x, const double a){
    vec yabs = abs(y);
    vec signY = sign(y);
    project_split(yabs.memptr(), w.memptr(), x.memptr(), yabs.n_elem, a);
    x %= signY;
}

void test_projection(){
    int l = 100;
    vec y = abs(randn<vec>(l));
    vec x = zeros<vec>(l);
    vec w = abs(randn<vec>(l));
    double a = 1;
    double epsilon = 1e-7;
    project_bucket( y.memptr(), w.memptr(), x.memptr(), l, a);
    double r = dot(w,x);
    if((r - a) > epsilon){
        cout << r << endl;
        printf("bucket failed \n");
    }
    project_bucket_filter( y.memptr(), w.memptr(), x.memptr(), l, a);
    r = dot(w,x);
    if((r - a) > epsilon){
        cout << r << endl;
        printf("bucket filter failed \n");
    }
    project_sort( y.memptr(), w.memptr(), x.memptr(), l, a);
    r = dot(w,x);
    if((r - a) > epsilon){
        cout << r << endl;
        printf("sort failed \n");
    }
    project_split( y.memptr(), w.memptr(), x.memptr(), l, a);
    r = dot(w,x);
    if((r - a) > epsilon){
        cout << r << endl;
        printf("split failed \n");
    }
}

}  // namespace proj
