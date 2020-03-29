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
#include "Solver.hpp"
#include "Projection.hpp"

using namespace std;
using namespace arma;



class SolverAXBWeightedProj: public SolverAXB
{
public:
    SolverAXBWeightedProj(mat A_, vec b_, vec x0, SolverConfiguration& cfg_, vec w_, double a_):
        SolverAXB(A_, b_, x0, cfg_),
        w(w_),
        a(a_),
        xp(x0.size()),
        norms(new double[cfg.solve_iter_max]{0}),
        step_sizes(new double[cfg.solve_iter_max]{0})
        {}
    
    ~SolverAXBWeightedProj(){
        delete norms;
        delete step_sizes;
    }

    virtual void solve(){
        converged = false;
        
        project(x, w, x, a);               // put initial point into the ball, if not already there
        r = b-(A*x);
        cur_norm = norm(r);
        solve_iter = 0;
        while(!converged){                                                      // solving iterations
            prev_norm = cur_norm;
            step_size = 1.;
            mat grad = -A.t()*(b-(A*x)) / sum(square(A)).t();                   // Gradient value
            xp = x - step_size * grad;                                          // Gradient step
            project(xp, w, xp, a);                                              // Projection
            r = b-A*xp;
            cur_norm = norm(r);  
            ls_iter = 0;                                                        // Current reconstruction value
            while (cur_norm > prev_norm && ls_iter < cfg.ls_iter_max){          // Line search (for the step size)
                step_size /= cfg.step_decrease_factor;
                xp = x - step_size * grad;                                      // Gradient step
                project(xp, w, xp, a);                                          // Projection
                r = b-A*xp;
                cur_norm = norm(r);
                ++ls_iter;
            }
            x = xp;
            norms[solve_iter] = cur_norm;
            step_sizes[solve_iter] = step_size;

            ++solve_iter;
            if (solve_iter > cfg.solve_iter_max 
                    || abs(cur_norm - prev_norm) < cfg.min_loss_change){
                converged = true;
            }
        }
    }
    void reset(vec x_, vec w_, double a_){
            x = x_;
            w = w_;
            a = a_;
        }

    bool converged;
    int ls_iter;
    int solve_iter;
    double a;
    double cur_norm;
    double prev_norm;
    double step_size;
    double * norms;
    double * step_sizes;
    vec xp;
    vec w;
    vec r;
};

