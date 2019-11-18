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
#include "Bench.hpp"

using namespace std;
using namespace arma;

 
int main(int argc, char **argv){ 
    test_projection();
    arma_rng::set_seed(2);         // set the seed
   	int k = 15;                     // numbr of non-zeros component of x
    int n = 100;                    // number of rows
    int m = 256;                    // number of columns (should be greater than n)
    double a = k;                   // radius
    int nbQ = 4;                    // Number of q values for lq
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
    cfg.epsilonQ = 1e-7;
    cfg.min_loss_change = 1e-5;
    cfg.solve_iter_max = 1000;
    cfg.solve_timeout = 60;
    cfg.step_decrease_factor = 2.;
    cfg.min_reweight_change = 1e-8;

    
    vec x = randn<vec>(m);


    // Iterative lQ projection
    
    BenchResults * res_lasso = bench_Lasso(A, b, x, cfg, a);
    BenchResults * res_candes = bench_candes(A, b, x, cfg, a);
    BenchResults * res_lq3 = bench_LQ(A, b, x, cfg, a, 3);
    BenchResults * res_lq4 = bench_LQ(A, b, x, cfg, a, 4);
    BenchResults * res_lq5 = bench_LQ(A, b, x, cfg, a, 5);
    BenchResults * res_lq6 = bench_LQ(A, b, x, cfg, a, 6);

    cout << "Lasso"<<endl;
    res_lasso->print();
    cout << "Candes"<<endl;
    res_candes->print();
    cout << "LQ3"<<endl;
    res_lq3->print();
    cout << "LQ4"<<endl;
    res_lq4->print();
    cout << "LQ5"<<endl;
    res_lq5->print();
    cout << "LQ6"<<endl;
    res_lq6->print();


    return 0;
}

