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

namespace solverAxb {

BenchResults bench_Lasso(arma::mat A, arma::vec b, arma::vec x,
                         SolverConfiguration& cfg, double a) {
  auto start = std::chrono::system_clock::now();
  SolverAXBProj slvrw(A, b, x, cfg, a);
  slvrw.solve();
  auto end = std::chrono::system_clock::now();

  BenchResults res;
  res.L0 = arma::accu(arma::abs(slvrw.x) >= cfg.epsilon);
  res.L1 = arma::norm(slvrw.x, 1);
  res.lossrec = slvrw.norms[slvrw.solve_iter - 1];
  res.elapsed_seconds = end - start;
  res.nbIteration = slvrw.solve_iter;
  res.cfg = cfg;
  res.n = A.n_cols;
  res.m = A.n_rows;
  res.a = a;
  res.nbQ = 0;
  return res;
}

}  // namespace solverAxb
