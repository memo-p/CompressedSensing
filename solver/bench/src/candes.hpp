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

#include "benchResults.hpp"

BenchResults bench_candes(arma::mat A, arma::vec b, arma::vec x,
                          SolverConfiguration cfg, double a) {
  auto start = std::chrono::system_clock::now();
  cfg.epsilonQ = 0.1;
  arma::vec w = arma::ones<arma::vec>(x.n_elem);
  SolverAXBWeightedProj slvrw(A, b, x, cfg, w, a);
  slvrw.solve();  // One iteration of Lasso             (init with l1 sol)
  SolverAXBWPLQFixe slvrwqf(A, b, slvrw.x, cfg, w, a, 0);
  slvrwqf.solve();  // One iteration with a epsilon = 1e-1 (fast search stage)
  cfg.epsilonQ = 0.01;
  slvrwqf.solve();  // One iteration with a epsilon = 1e-2 (final optimization
                    // stage)
  auto end = std::chrono::system_clock::now();

  BenchResults res;
  res.L0 = accu(abs(slvrwqf.x) >= cfg.epsilon);
  res.L1 = norm(slvrwqf.x, 1);
  res.lossrec = slvrwqf.norms[slvrwqf.solve_iter - 1];
  res.elapsed_seconds = end - start;
  res.nbIteration = slvrwqf.total_iter + slvrw.solve_iter;
  res.cfg = cfg;
  res.n = A.n_cols;
  res.m = A.n_rows;
  res.a = a;
  res.nbQ = 1;
  return res;
}
