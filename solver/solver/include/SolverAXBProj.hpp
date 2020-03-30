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
#include <vector>

#include "Solver.hpp"
#include "projection.hpp"

namespace solverAxb {

class SolverAXBProj : public SolverAXB {
 public:
  bool converged;
  int ls_iter;
  int solve_iter;
  double a;
  double cur_norm;
  double prev_norm;
  double step_size;
  arma::vec r;
  arma::vec xp;
  double* norms;
  double* step_sizes;

  SolverAXBProj(arma::mat A_, arma::vec b_, arma::vec x0,
                SolverConfiguration& cfg_, double a_)
      : SolverAXB(A_, b_, x0, cfg_),
        a(a_),
        xp(x0.size()),
        norms(new double[cfg.solve_iter_max]{0}),
        step_sizes(new double[cfg.solve_iter_max]{0}) {}

  virtual void solve() {
    converged = false;
    // put initial point into the ball, if not already there
    proj::project(x, x, a);
    r = b - (A * x);
    cur_norm = arma::norm(r);
    solve_iter = 0;
    while (!converged) {  // solving iterations
      prev_norm = cur_norm;
      step_size = 1.;
      arma::mat grad = -A.t() * (b - (A * x)) /
                       arma::sum(arma::square(A)).t();  // Gradient value
      xp = x - step_size * grad;                        // Gradient step
      proj::project(xp, xp, a);                         // Projection
      r = b - A * xp;
      cur_norm = arma::norm(r);
      ls_iter = 0;  // Current reconstruction value
      while (cur_norm > prev_norm &&
             ls_iter < cfg.ls_iter_max) {  // Line search (for the step size)
        step_size /= cfg.step_decrease_factor;
        xp = x - step_size * grad;  // Gradient step
        proj::project(xp, xp, a);   // Projection
        r = b - A * xp;
        cur_norm = arma::norm(r);
        ++ls_iter;
      }
      x = xp;
      norms[solve_iter] = cur_norm;
      step_sizes[solve_iter] = step_size;

      ++solve_iter;
      if (solve_iter > cfg.solve_iter_max ||
          std::abs(cur_norm - prev_norm) < cfg.min_loss_change) {
        converged = true;
      }
    }
  }

  void reset(arma::vec x_, double a_) {
    x = x_;
    a = a_;
  }
};

}  // namespace solverAxb
