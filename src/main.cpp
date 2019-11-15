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
 
#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>
#include <armadillo>
#include "SolverAXBWeightedProj.hpp"
#include "SolverAXBWPLQFixe.hpp"
#include "SolverAXBWPLQ.hpp"

using namespace std;
using namespace arma;

 
int main(int argc, char **argv){
    arma_rng::set_seed(2);         // set the seed
   	int k = 15;                     // numbr of non-zeros component of x
    int n = 100;                    // number of rows
    int m = 256;                    // number of columns (should be greater than n)
    double a = k;                   // radius
    mat A = randn<mat>(n,m);        // Matrix A
    vec x0 = zeros<vec>(m);         // true x (with sparsity k) and gaussian values
    for (int i = 0; i < k; ++i){
        int id = rand() % m;
        x0[id] = randn(1)[0];
    }
    vec b = A*x0; 

    SolverConfiguration cfg;
    cfg.ls_iter_max = 30;
    cfg.epsilon = 1e-8;
    cfg.epsilonQ = 1e-8;
    cfg.min_loss_change = 1e-8;
    cfg.solve_iter_max = 1000;
    cfg.solve_timeout = 60;
    cfg.step_decrease_factor = 2.;
    cfg.min_reweight_change = 1e-8;

    vec w = ones<vec>(m);
    vec x = randn<vec>(m);

    // Lasso 
    SolverAXBWeightedProj slvrw(A, b, x, cfg, w, a);
    slvrw.solve();
    cout << "LASSO method" << endl;
    cout << "norm L0 of x " << accu(abs(slvrw.x)>= cfg.epsilon) << endl;
    cout << "Reconstruction " << slvrw.norms[slvrw.solve_iter-1]  << endl;
    cout << endl;

    // Direct use of L0 via reweighting
    SolverAXBWPLQFixe slvrwqf(A, b, slvrw.x, cfg, w, a, 0);
    slvrwqf.solve();
    cout << "L0 reweighting method" << endl;
    cout << "norm L0 of x " << accu(abs(slvrwqf.x)>= cfg.epsilon) << endl;
    cout << "Reconstruction " << slvrwqf.norms[slvrwqf.solve_iter-1]  << endl;
    cout << endl;

    // Iterative lQ projection
    int nbQ = 11;
    SolverAXBWPLQ slvrwq(A, b, x, cfg, a, 11);
    slvrwq.solve();
    cout << "Iterative L0 reweighting method" << endl;
    cout << "norm L0 of x " << accu(abs(slvrwq.x)>= cfg.epsilon) << endl;
    cout << "Reconstruction " << slvrwq.recNorms[nbQ-1]  << endl;
    cout << endl;

    // for (size_t i = 0; i < nbQ; i++)
    // {
    //     printf("%.2e;%.0f;%.2f;%.0f \n",slvrwq.recNorms[i], slvrwq.normsXL0[i], slvrwq.normsXL1[i], slvrwq.nbIters[i]);
    // }
    

    
    test_projection();

    return 0;
}

