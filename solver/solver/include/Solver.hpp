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

#include <armadillo>

namespace solverAxb {

class SolverConfiguration {
 public:
  int solve_iter_max;
  int solve_timeout;
  int ls_iter_max;
  double step_decrease_factor;
  double min_loss_change;
  double min_reweight_change;
  double epsilon;
  double epsilonQ;
};

class SolverAXB {
 public:
  SolverAXB(arma::mat A_, arma::vec b_, arma::vec x0, SolverConfiguration& cfg_)
      : A(A_), b(b_), x(x0), cfg(cfg_) {}

  virtual void solve() = 0;

  arma::mat A;
  arma::vec b;
  arma::vec x;
  SolverConfiguration& cfg;
};

}  // namespace solverAxb
