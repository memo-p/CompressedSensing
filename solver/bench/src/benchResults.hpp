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

#include "SolverAXBWPLQ.hpp"

struct BenchResults {
  double L0;
  double L1;
  double lossrec;
  std::chrono::duration<double> elapsed_seconds;
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