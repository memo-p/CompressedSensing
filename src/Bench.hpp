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
#include "SolverAXBWPLQ.hpp"

using namespace std;
using namespace arma;

struct BenchResults{
    double L0;
    double L1;
    double lossrec;
    chrono::duration<double> elapsed_seconds;
    int nbIteration;

    void print(){
        printf("L0  = %.0f \n", L0);
        printf("L1  = %.2f \n", L1);
        printf("rec = %.2f \n", lossrec);
        printf("time= %.4f s\n", elapsed_seconds.count());
        printf("#it = %d \n", nbIteration);
        
    }
};



BenchResults * bench_Lasso(mat A, vec b, vec x, SolverConfiguration& cfg, double a){
    vec w = ones<vec>(x.n_elem);
    auto start = std::chrono::system_clock::now();
    SolverAXBWeightedProj slvrw(A, b, x, cfg, w, a);
    slvrw.solve();
    auto end = std::chrono::system_clock::now();

    BenchResults * res = new BenchResults();
    res->L0 = accu(abs(slvrw.x)>= cfg.epsilon) ;
    res->L1 = norm(slvrw.x, 1);
    res->lossrec = slvrw.norms[slvrw.solve_iter-1] ;
    res->elapsed_seconds = end-start;
    res->nbIteration = slvrw.solve_iter;
    cout << endl;
    return res;
}



BenchResults * bench_candes(mat A, vec b, vec x, SolverConfiguration cfg, double a){
    auto start = chrono::system_clock::now();
    cfg.epsilonQ = 0.1;
    vec w = ones<vec>(x.n_elem);
    SolverAXBWeightedProj slvrw(A, b, x, cfg, w, a);            
    slvrw.solve();                                              // One iteration of Lasso             (init with l1 sol)
    SolverAXBWPLQFixe slvrwqf(A, b, slvrw.x, cfg, w, a, 0);     
    slvrwqf.solve();                                            // One iteration with a epsilon = 1e-1 (fast search stage)
    cfg.epsilonQ = 0.01;
    slvrwqf.solve();                                            // One iteration with a epsilon = 1e-2 (final optimization stage)
    auto end = std::chrono::system_clock::now();

    BenchResults * res = new BenchResults();
    res->L0 = accu(abs(slvrwqf.x)>= cfg.epsilon);
    res->L1 = norm(slvrwqf.x, 1);
    res->lossrec = slvrwqf.norms[slvrwqf.solve_iter-1] ;
    res->elapsed_seconds = end-start;
    res->nbIteration = slvrwqf.total_iter + slvrw.solve_iter;
    return res;
}



BenchResults * bench_LQ(mat A, vec b, vec x, SolverConfiguration cfg, double a, int nbQ = 4){

    auto start = chrono::system_clock::now();
    SolverAXBWPLQ slvrwq(A, b, x, cfg, a, nbQ);
    slvrwq.solve();
    auto end = std::chrono::system_clock::now();
    
    BenchResults * res = new BenchResults();
    res->nbIteration  = 0;
    for (size_t i = 0; i < nbQ; i++){
        res->nbIteration += slvrwq.nbIters[i];
    }
    res->L0 = slvrwq.normsXL0[nbQ-1];
    res->L1 = slvrwq.normsXL1[nbQ-1];
    res->lossrec = slvrwq.recNorms[nbQ-1] ;
    res->elapsed_seconds = end-start;
    return res;
}

