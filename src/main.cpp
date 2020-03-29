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

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include "Bench.hpp"

using namespace std;
using namespace arma;

int main(int argc, char **argv) {
  test_projection();
  arma_rng::set_seed(5);     // set the seed
  int k = 30;                // number of non-zeros component of x
  int n = 100;               // number of rows
  int m = 256;               // number of columns (should be greater than n)
  double a = k;              // radius
  int nbQ = 4;               // Number of q values for lq
  mat A = randn<mat>(n, m);  // Matrix A
  vec x0 = zeros<vec>(m);    // true x (with sparsity k) and gaussian values
  for (int i = 0; i < k; ++i) {
    int id = rand() % m;
    x0[id] = randn(1)[0];
  }
  vec b = A * x0;

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

  analyse_different_algorithms(A, b, x, cfg, a);

//   analyse_LQ_fct_nbQ(A, b, x, cfg, a);

  // analyse_LQ_fct_nbQ_by_iter(A, b, x, cfg, a);

//   analyse_LQ_fct_radius(A, b, x, cfg, k * .5, k * 2, 20);

  return 0;
}
