
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

BenchResults bench_LQ(arma::mat A, arma::vec b, arma::vec x,
                      SolverConfiguration cfg, double a, int nbQ = 4) {
  auto start = std::chrono::system_clock::now();
  SolverAXBWPLQ slvrwq(A, b, x, cfg, a, nbQ);
  slvrwq.solve();
  auto end = std::chrono::system_clock::now();

  BenchResults res;
  res.nbIteration = 0;
  for (int i = 0; i < nbQ; i++) {
    res.nbIteration += slvrwq.nbIters[i];
  }
  res.L0 = slvrwq.normsXL0[nbQ - 1];
  res.L1 = slvrwq.normsXL1[nbQ - 1];
  res.lossrec = slvrwq.recNorms[nbQ - 1];
  res.elapsed_seconds = end - start;
  res.cfg = cfg;
  res.n = A.n_cols;
  res.m = A.n_rows;
  res.a = a;
  res.nbQ = nbQ;
  return res;
}