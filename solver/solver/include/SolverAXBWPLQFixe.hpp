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

#include "SolverAXBWeightedProj.hpp"

namespace solverAxb {

class SolverAXBWPLQFixe : public SolverAXB {
 public:
  SolverAXBWeightedProj slvrwp;
  arma::vec w;
  arma::vec r;
  double q;
  double a;
  double* norms;
  double* normsX;
  double cur_norm;
  double cur_normX;
  double prev_norm;
  int solve_iter;
  int total_iter;
  bool converged;
  SolverAXBWPLQFixe(arma::mat A_, arma::vec b_, arma::vec x0,
                    SolverConfiguration& cfg_, arma::vec w_, double a_,
                    double q_)
      : SolverAXB(A_, b_, x0, cfg_),
        slvrwp(A_, b_, x0, cfg_, w_, a_),
        w(w_),
        q(q_),
        a(a_),
        norms(new double[cfg.solve_iter_max]{0}),
        normsX(new double[cfg.solve_iter_max]{0}) {
    total_iter = 0;
  }

  virtual void solve() {
    converged = false;
    solve_iter = 0;
    r = b - (A * x);
    cur_norm = arma::norm(r);
    while (!converged) {
      prev_norm = cur_norm;
      w = 1. /
          arma::pow(arma::abs(x + cfg.epsilonQ), 1 - q);  // Update the weight
      slvrwp.reset(x, w, a);  // Solve for the current weight
      slvrwp.solve();

      r = b - (A * slvrwp.x);
      cur_norm = arma::norm(r);
      cur_normX = arma::norm(x - slvrwp.x);
      norms[solve_iter] = cur_norm;
      normsX[solve_iter] = cur_normX;

      ++solve_iter;
      x = slvrwp.x;
      total_iter += slvrwp.solve_iter;

      if (cur_normX < cfg.min_reweight_change ||
          solve_iter > cfg.solve_iter_max) {
        converged = true;
      }
    }
    total_iter += solve_iter;
  }
};

}  // namespace solverAxb
