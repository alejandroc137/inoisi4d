#ifndef PARAM
#define PARAM

#include <stddef.h>
#include <gsl/gsl_rng.h>

extern const char model_name[];

extern const double param_x0start;
extern const double param_x0end;
extern const double param_x1start;
extern const double param_x1end;
extern const double param_x2start;
extern const double param_x2end;
extern const double param_x3start;
extern const double param_x3end;

void param_read_params(char* filename);

void param_write_params(char* filename);

void param_set_output_name(char* filename, size_t filename_size,
                           int ni, int nj, int nk, int nl,
                           int npi, int npj, int npk, int npl, char* dir);

double param_env(double raw, double avg_raw, double var_raw,
		 int i, int j, int k, int l, int ni, int nj, int nk, int nl,
		 int pi, int pj, int pk, int pl, double dx0, double dx1, double dx2, double dx3);

void param_coeff(double* coeff, double x0, double x1, double x2, double x3, double dx0,
		 double dx1, double dx2, double dx3, int index);

void param_set_source(double* values, gsl_rng* rstate, int ni, int nj, int nk, int nl,
		      int pi, int pj, int pk, int pl, int npi, int npj, int npk, int npl,
		      double dx0, double dx1, double dx2, double dx3, int nrecur);

#endif
