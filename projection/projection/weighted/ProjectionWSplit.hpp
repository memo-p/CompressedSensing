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

#include <iostream>
#include <cstdio>
#include <armadillo>

using namespace arma;
using namespace std;

#define KahanSum(s,v,c,t,y) y = v - c; t = s + y; c = (t - s) - y; s = t;
#define KahanSumDel(s,v,c,t,y) y = v + c; t = s - y; c = (t - s) + y; s = t;


namespace proj {

void project_split(double* y, double* w,  double* x,
const unsigned int length, const double a)
{	
	double*	aux = (double*)malloc(length*sizeof(double));
	double*  aux0 = aux;
	double*	waux = (double*)malloc(length*sizeof(double));
	double*  waux0 = waux;
	int		auxlength=1; 
	int		auxlengthold=-1;	
	double sumWY = *w * (*aux=*y);
	double csumWY = 0;
	double tsumWY = 0;
	double ysumWY = 0;
	double Ws = (*waux=*w) * *w;
	double cWs = 0;
	double tWs = 0;
	double yWs = 0;
	double	tau = (sumWY - a) / Ws;
	int 	i=1;
	int iter = 1;
	for (; i<length; i++) 
		if (y[i] > w[i] * tau) {
			aux[auxlength]=y[i];
			KahanSum(sumWY,y[i]*w[i],csumWY,tsumWY,ysumWY);
			// sumWY += (aux[auxlength]=y[i])*w[i];
			waux[auxlength]=w[i];
			KahanSum(Ws,w[i]*w[i],cWs,tWs,yWs);
			// Ws += (waux[auxlength]=w[i])*w[i];
			tau = (sumWY - a) / Ws;
			if (tau < (w[i]*y[i]-a)/(w[i]*w[i])) {
				csumWY = 0;
				sumWY = 0;
				KahanSum(sumWY,y[i]*w[i],csumWY,tsumWY,ysumWY);
				// sumWY = w[i]*y[i];
				cWs = 0;
				Ws = 0;
				KahanSum(Ws,w[i]*w[i],cWs,tWs,yWs);
				// Ws = w[i]*w[i];
				tau = (sumWY - a) / Ws;
				auxlengthold=auxlength-1;
			}
			auxlength++;
		} 
	if (auxlengthold>=0) {
		iter++;
		auxlength -= ++auxlengthold;
		aux  += auxlengthold;
		waux += auxlengthold;
		while (--auxlengthold>=0) 
			if (aux0[auxlengthold] > waux0[auxlengthold] * tau) {
				*(--aux) = aux0[auxlengthold];
				KahanSum(sumWY,aux0[auxlengthold] * waux0[auxlengthold],csumWY,tsumWY,ysumWY);
				// sumWY += (*(--aux)=aux0[auxlengthold]) * waux0[auxlengthold];
				*(--waux) = waux0[auxlengthold];
				KahanSum(Ws,waux0[auxlengthold] * waux0[auxlengthold],cWs,tWs,yWs);
				// Ws += (*(--waux)=waux0[auxlengthold]) * waux0[auxlengthold];
				tau = (sumWY - a) / Ws;
				auxlength++;					
			}
	}
	do {
		iter++;
		auxlengthold=auxlength-1;
		for (i=auxlength=0; i<=auxlengthold; i++)
			if (aux[i] > waux[i] * tau) {
				aux[auxlength]=aux[i];	
				waux[auxlength++]=waux[i];	
			}
			else{
				KahanSumDel(sumWY,aux[i] * waux[i],csumWY,tsumWY,ysumWY);
				// sumWY -= aux[i] * waux[i];
				KahanSumDel(Ws,waux[i] * waux[i],cWs,tWs,yWs);
				// Ws -= waux[i] * waux[i];
				tau = (sumWY - a) / Ws;
			} 
	} while (auxlength<=auxlengthold);
// printf("%f\n",tau);
// printf("%d\n",iter);
// 	sumWY = Ws = 0;
// 	cWs = csumWY = 0;
// 	for (int i = 0; i <= auxlengthold; ++i)
// 	{
// 		KahanSum(sumWY, aux[i] * waux[i],csumWY,tsumWY,ysumWY);
// 		// sumWY += aux[i] * waux[i];
// 		KahanSum(Ws,waux[i] * waux[i],cWs,tWs,yWs);
// 		// Ws += waux[i] * waux[i];
// 		// printf("w[%d]=%f\n",i,waux[i]);
// 	}
// 	tau = (sumWY - a) / Ws;

	for (i=0; i<length; i++)
		x[i] = (y[i] > w[i]*tau ? y[i] - w[i] * tau : 0.0); 
	free(aux0); 
	free(waux0);
} 

}  // namespace proj

