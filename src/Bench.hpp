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

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include "SolverAXBWPLQ.hpp"

using namespace std;
using namespace arma;

struct BenchResults {
  double L0;
  double L1;
  double lossrec;
  chrono::duration<double> elapsed_seconds;
  int nbIteration;
  SolverConfiguration cfg;
  int n;
  int m;
  double a;
  int nbQ;

  void print() {
    printf("L0  = %.0f \n", L0);
    printf("L1  = %.2f \n", L1);
    printf("rec = %.2f \n", lossrec);
    printf("time= %.4f s\n", elapsed_seconds.count());
    printf("#it = %d \n", nbIteration);
    printf("n = %d \n", n);
    printf("m = %d \n", m);
    printf("a = %.2f \n", a);
    printf("#q = %d \n", nbQ);
  }
  void printCSV() {
    printf("%.0f;", L0);
    printf("%.2f;", L1);
    printf("%.2f;", lossrec);
    printf("%.4f;", elapsed_seconds.count());
    printf("%d;", nbIteration);
    printf("%d;", n);
    printf("%d;", m);
    printf("%.2f;", a);
    printf("%d;", nbQ);
    printf("\n");
  }
  void printCSVHead() {
    printf("L0;");
    printf("L1;");
    printf("rec;");
    printf("time;");
    printf("#it;");
    printf("n;");
    printf("m;");
    printf("a;");
    printf("nbQ;");
    printf("\n");
  }
};

BenchResults* bench_Lasso(mat A, vec b, vec x, SolverConfiguration& cfg,
                          double a) {
  vec w = ones<vec>(x.n_elem);
  auto start = std::chrono::system_clock::now();
  SolverAXBWeightedProj slvrw(A, b, x, cfg, w, a);
  slvrw.solve();
  auto end = std::chrono::system_clock::now();

  BenchResults* res = new BenchResults();
  res->L0 = accu(abs(slvrw.x) >= cfg.epsilon);
  res->L1 = norm(slvrw.x, 1);
  res->lossrec = slvrw.norms[slvrw.solve_iter - 1];
  res->elapsed_seconds = end - start;
  res->nbIteration = slvrw.solve_iter;
  res->cfg = cfg;
  res->n = A.n_cols;
  res->m = A.n_rows;
  res->a = a;
  res->nbQ = 0;
  cout << endl;
  return res;
}

BenchResults* bench_candes(mat A, vec b, vec x, SolverConfiguration cfg,
                           double a) {
  auto start = chrono::system_clock::now();
  cfg.epsilonQ = 0.1;
  vec w = ones<vec>(x.n_elem);
  SolverAXBWeightedProj slvrw(A, b, x, cfg, w, a);
  slvrw.solve();  // One iteration of Lasso             (init with l1 sol)
  SolverAXBWPLQFixe slvrwqf(A, b, slvrw.x, cfg, w, a, 0);
  slvrwqf.solve();  // One iteration with a epsilon = 1e-1 (fast search stage)
  cfg.epsilonQ = 0.01;
  slvrwqf.solve();  // One iteration with a epsilon = 1e-2 (final optimization
                    // stage)
  auto end = std::chrono::system_clock::now();

  BenchResults* res = new BenchResults();
  res->L0 = accu(abs(slvrwqf.x) >= cfg.epsilon);
  res->L1 = norm(slvrwqf.x, 1);
  res->lossrec = slvrwqf.norms[slvrwqf.solve_iter - 1];
  res->elapsed_seconds = end - start;
  res->nbIteration = slvrwqf.total_iter + slvrw.solve_iter;
  res->cfg = cfg;
  res->n = A.n_cols;
  res->m = A.n_rows;
  res->a = a;
  res->nbQ = 1;
  return res;
}

BenchResults* bench_LQ(mat A, vec b, vec x, SolverConfiguration cfg, double a,
                       int nbQ = 4) {
  auto start = chrono::system_clock::now();
  SolverAXBWPLQ slvrwq(A, b, x, cfg, a, nbQ);
  slvrwq.solve();
  auto end = std::chrono::system_clock::now();

  BenchResults* res = new BenchResults();
  res->nbIteration = 0;
  for (size_t i = 0; i < nbQ; i++) {
    res->nbIteration += slvrwq.nbIters[i];
  }
  res->L0 = slvrwq.normsXL0[nbQ - 1];
  res->L1 = slvrwq.normsXL1[nbQ - 1];
  res->lossrec = slvrwq.recNorms[nbQ - 1];
  res->elapsed_seconds = end - start;
  res->cfg = cfg;
  res->n = A.n_cols;
  res->m = A.n_rows;
  res->a = a;
  res->nbQ = nbQ;
  return res;
}

BenchResults** bench_LQ_fct_nbQ(mat A, vec b, vec x, SolverConfiguration cfg,
                                double a, int nbQ_min = 2, int nbQ_max = 10) {
  int nb_bench = nbQ_max - nbQ_min + 1;
  BenchResults** res = new BenchResults*[nb_bench];
  for (int i = 0; i < nb_bench; i++) {
    res[i] = bench_LQ(A, b, x, cfg, a, i + nbQ_min);
  }
  return res;
}

void analyse_LQ_fct_nbQ(mat A, vec b, vec x, SolverConfiguration cfg,
                        double a) {
  int nbQ_min = 1;
  int nbQ_max = 10;
  int nbQs = nbQ_max - nbQ_min + 1;
  BenchResults** res = bench_LQ_fct_nbQ(A, b, x, cfg, a, nbQ_min, nbQ_max);
  res[0]->printCSVHead();
  for (int i = 0; i < nbQs; ++i) {
    res[i]->printCSV();
  }
}

BenchResults** bench_LQ_fct_radius(mat A, vec b, vec x, SolverConfiguration cfg,
                                   double a_min, double a_max, uint64_t nb_a,
                                   uint64_t q) {
  int nb_bench = nb_a;
  double step = (a_max - a_min) / nb_a;
  BenchResults** res = new BenchResults*[nb_bench];
  for (int i = 0; i < nb_bench; i++) {
    res[i] = bench_LQ(A, b, x, cfg, a_min + i * step, q);
  }
  return res;
}

void analyse_LQ_fct_radius(mat A, vec b, vec x, SolverConfiguration cfg,
                           double a_min, double a_max, uint64_t nb_a,
                           uint64_t q = 4) {
  BenchResults** res = bench_LQ_fct_radius(A, b, x, cfg, a_min, a_max, nb_a, q);
  res[0]->printCSVHead();
  for (int i = 0; i < nb_a; ++i) {
    res[i]->printCSV();
  }
}

void analyse_different_algorithms(mat A, vec b, vec x, SolverConfiguration cfg,
                                  double a) {
  // Iterative lQ projection

  BenchResults* res_lasso = bench_Lasso(A, b, x, cfg, a);
  BenchResults* res_candes = bench_candes(A, b, x, cfg, a);
  BenchResults* res_lq3 = bench_LQ(A, b, x, cfg, a, 3);
  BenchResults* res_lq4 = bench_LQ(A, b, x, cfg, a, 4);
  BenchResults* res_lq5 = bench_LQ(A, b, x, cfg, a, 5);
  BenchResults* res_lq6 = bench_LQ(A, b, x, cfg, a, 6);

  cout << "Lasso" << endl;
  res_lasso->print();
  cout << "Candes" << endl;
  res_candes->print();
  cout << "LQ3" << endl;
  res_lq3->print();
  cout << "LQ4" << endl;
  res_lq4->print();
  cout << "LQ5" << endl;
  res_lq5->print();
  cout << "LQ6" << endl;
  res_lq6->print();

  cout << "Algo;";
  res_lasso->printCSVHead();
  cout << "Lasso;";
  res_lasso->printCSV();
  cout << "Candes;";
  res_candes->printCSV();
  cout << "LQ3;";
  res_lq3->printCSV();
//   cout << "LQ4" ;
//   res_lq4->printCSV();
  cout << "LQ5;";
  res_lq5->printCSV();
//   cout << "LQ6" << endl;
//   res_lq6->printCSV();
}

void analyse_LQ_fct_nbQ_by_iter(mat A, vec b, vec x, SolverConfiguration cfg,
                                double a, int nbQ = 3) {
  SolverAXBWPLQ slvrwq(A, b, x, cfg, a, nbQ);
  slvrwq.solve();

  printf("nbIters=[");
  for (int i = 0; i < nbQ; ++i) {
    printf("%d, ", slvrwq.nbIters[i]);
  }
  printf("]\n");

  printf("L0s=[");
  for (int i = 0; i < nbQ; ++i) {
    printf("%.0f, ", slvrwq.normsXL0[i]);
  }
  printf("]\n");

  printf("L1s=[");
  for (int i = 0; i < nbQ; ++i) {
    printf("%.5f, ", slvrwq.normsXL1[i]);
  }
  printf("]\n");

  printf("recs=[");
  for (int i = 0; i < nbQ; ++i) {
    printf("%f, ", slvrwq.recNorms[i]);
  }
  printf("]\n");

  vec q = reverse(linspace(0., 1, nbQ));
  printf("q=[");
  for (int i = 0; i < nbQ; ++i) {
    printf("%.5f, ", q[i]);
  }
  printf("]\n");
}