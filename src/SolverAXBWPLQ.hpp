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
#include "SolverAXBWPLQFixe.hpp"

using namespace std;
using namespace arma;

class SolverAXBWPLQ: public SolverAXB
{
private:
    
public:
    SolverAXBWPLQ(mat A_, vec b_, vec x0, SolverConfiguration& cfg_, double a_, int nbQ_):
        SolverAXB(A_, b_, x0, cfg_),
        slvrwpq(A_, b_, x0, cfg_,  ones<vec>(x0.n_elem),  a_, 1),
        a(a_),
        nbQ(nbQ_),
        q(nbQ_),
        recNorms(new double[nbQ_]{0}),
        normsXL0(new double[nbQ_]{0}),
        normsXL1(new double[nbQ_]{0}),
        nbIters(new double[nbQ_]{0})
        {}
    
    virtual void solve(){
        q = reverse(linspace(0., 1, nbQ));
        
        for (size_t i = 0; i < nbQ; i++){
            slvrwpq.q = q[i];
            slvrwpq.solve();
            recNorms[i] = slvrwpq.norms[slvrwpq.solve_iter-1] ;
            normsXL0[i] = accu(abs(slvrwpq.x) >= cfg.epsilon);
            normsXL1[i] = norm(slvrwpq.x,1);
            nbIters[i] = slvrwpq.total_iter;
        }
        x = slvrwpq.x;
    }

    int nbQ;
    double a;
    double * recNorms;
    double * normsXL0;
    double * normsXL1;
    double * nbIters;
    vec q;
    SolverAXBWPLQFixe slvrwpq;
};

