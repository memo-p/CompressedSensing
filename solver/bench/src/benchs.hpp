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

#include "lq.hpp"

namespace solverAxb {

std::vector<BenchResults> bench_LQ_fct_nbQ(arma::mat A, arma::vec b,
                                           arma::vec x, SolverConfiguration cfg,
                                           double a, int nbQ_min = 2,
                                           int nbQ_max = 10) {
  int nb_bench = nbQ_max - nbQ_min + 1;
  std::vector<BenchResults> res(nb_bench);
  for (int i = 0; i < nb_bench; i++) {
    res[i] = bench_LQ(A, b, x, cfg, a, i + nbQ_min);
  }
  return res;
}

void analyse_LQ_fct_nbQ(arma::mat A, arma::vec b, arma::vec x,
                        SolverConfiguration cfg, double a) {
  int nbQ_min = 1;
  int nbQ_max = 10;
  int nbQs = nbQ_max - nbQ_min + 1;
  std::vector<BenchResults> res =
      bench_LQ_fct_nbQ(A, b, x, cfg, a, nbQ_min, nbQ_max);
  res[0].printCSVHead();
  for (int i = 0; i < nbQs; ++i) {
    res[i].printCSV();
  }
}

std::vector<BenchResults> bench_LQ_fct_radius(arma::mat A, arma::vec b,
                                              arma::vec x,
                                              SolverConfiguration cfg,
                                              double a_min, double a_max,
                                              uint64_t nb_a, uint64_t q) {
  int nb_bench = nb_a;
  double step = (a_max - a_min) / nb_a;
  std::vector<BenchResults> res(nb_bench);
  for (int i = 0; i < nb_bench; i++) {
    res[i] = bench_LQ(A, b, x, cfg, a_min + i * step, q);
  }
  return res;
}

}  // namespace solverAxb
