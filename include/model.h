#ifndef MODEL
#define MODEL

#include "HYPRE_sstruct_ls.h"

int model_set_gsl_seed(int seed, int myid);

void model_set_spacing(double* dx0, double* dx1, double* dx2, double* dx3,
		       int ni, int nj, int nk, int nl, int npi, int npj, int npk, int npl);

void model_set_periodic(long long int* bound, int ni, int nj, int nk, int nl,
			int npi, int npj, int npk, int npl, int dim);

void model_create_stencil(HYPRE_SStructStencil* stencil, int dim, int var);

void model_set_stencil_values(HYPRE_SStructMatrix* A, long long int* ilower, long long int* iupper,
			      int ni, int nj, int nk, int nl, int pi, int pj, int pk, int pl,
			      double dx0, double dx1, double dx2, double dx3, int part, int var);

void model_set_bound(HYPRE_SStructMatrix* A, int ni, int nj, int nk, int nl,
		     int pi, int pj, int pk, int pl, int npi, int npj, int npk, int npl,
		     double dx0, double dx1, double dx2, double dx3);

double model_area(int i, int j, int k, int l, int ni, int nj, int nk, int nl,
		  int pi, int pj, int pk, int pl, double dx0, double dx1, double dx2, double dx3);

#endif
