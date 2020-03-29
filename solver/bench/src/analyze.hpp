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

#include "benchs.hpp"
#include "candes.hpp"
#include "lasso.hpp"

void analyse_LQ_fct_radius(arma::mat A, arma::vec b, arma::vec x,
                           SolverConfiguration cfg, double a_min, double a_max,
                           uint64_t nb_a, uint64_t q = 4) {
  std::vector<BenchResults> res =
      bench_LQ_fct_radius(A, b, x, cfg, a_min, a_max, nb_a, q);
  res[0].printCSVHead();
  for (size_t i = 0; i < res.size(); ++i) {
    res[i].printCSV();
  }
}

void analyse_different_algorithms(arma::mat A, arma::vec b, arma::vec x,
                                  SolverConfiguration cfg, double a) {
  // Iterative lQ projection

  BenchResults res_lasso = bench_Lasso(A, b, x, cfg, a);
  BenchResults res_candes = bench_candes(A, b, x, cfg, a);
  BenchResults res_lq3 = bench_LQ(A, b, x, cfg, a, 3);
  BenchResults res_lq4 = bench_LQ(A, b, x, cfg, a, 4);
  BenchResults res_lq5 = bench_LQ(A, b, x, cfg, a, 5);
  BenchResults res_lq6 = bench_LQ(A, b, x, cfg, a, 6);

  std::cout << "Lasso" << std::endl;
  res_lasso.print();
  std::cout << "Candes" << std::endl;
  res_candes.print();
  std::cout << "LQ3" << std::endl;
  res_lq3.print();
  std::cout << "LQ4" << std::endl;
  res_lq4.print();
  std::cout << "LQ5" << std::endl;
  res_lq5.print();
  std::cout << "LQ6" << std::endl;
  res_lq6.print();

  std::cout << "Algo;";
  res_lasso.printCSVHead();
  std::cout << "Lasso;";
  res_lasso.printCSV();
  std::cout << "Candes;";
  res_candes.printCSV();
  std::cout << "LQ3;";
  res_lq3.printCSV();
  std::cout << "LQ4";
  res_lq4.printCSV();
  std::cout << "LQ5;";
  res_lq5.printCSV();
  std::cout << "LQ6" << std::endl;
  res_lq6.printCSV();
}

void analyse_LQ_fct_nbQ_by_iter(arma::mat A, arma::vec b, arma::vec x,
                                SolverConfiguration cfg, double a,
                                int nbQ = 3) {
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

  arma::vec q = arma::reverse(arma::linspace(0., 1, nbQ));
  printf("q=[");
  for (int i = 0; i < nbQ; ++i) {
    printf("%.5f, ", q[i]);
  }
  printf("]\n");
}