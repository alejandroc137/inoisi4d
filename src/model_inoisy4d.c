#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "_hypre_utilities.h"
#include "HYPRE_sstruct_ls.h"

#include "param.h"

#define NSTENCIL 33

int model_set_gsl_seed(int seed, int myid)
{
  return seed + myid;
}

void model_set_spacing(double* dx0, double* dx1, double* dx2, double* dx3,
		       int ni, int nj, int nk, int nl, int npi, int npj, int npk, int npl)
{
  *dx0 = (param_x0end - param_x0start) / (npk * nk);
  *dx1 = (param_x1end - param_x1start) / (npl * nl); 
  *dx2 = (param_x2end - param_x2start) / (npj * nj);
  *dx3 = (param_x3end - param_x3start) / (npi * ni);

  printf("dx0: %f \n",*dx0);
  printf("dx1: %f \n",*dx1);
  printf("dx2: %f \n",*dx2);
  printf("dx3: %f \n",*dx3);

}

/* Periodic in time only.  Spatial boundaries are non-periodic. */
void model_set_periodic(long long int* bound, int ni, int nj, int nk, int nl,
			int npi, int npj, int npk,int npl, int dim)
{
  (void) ni;
  (void) nj;
  (void) nl;
  (void) npi;
  (void) npj;
  (void) npl;
  (void) dim;

  bound[0] = 0;
  bound[1] = 0;
  bound[2] = 0;
  bound[3] = npk * nk;
}

void model_create_stencil(HYPRE_SStructStencil* stencil, int dim, int var)
{
  (void) dim;
  HYPRE_SStructStencilCreate(4, NSTENCIL, stencil);
  //i,j,l,k
  //z,y,x,t
  int entry;  
  long long int offsets[NSTENCIL][4] = {
    {0, 0, 0, 0},  // Center [0]
    {-1, 0, 0, 0}, {1, 0, 0, 0},  // -Deltax3, +Deltax3 [1, 2]
    {0, -1, 0, 0}, {0, 1, 0, 0},  // -Deltax2, +Deltax2 [3, 4]
    {0, 0, -1, 0}, {0, 0, 1, 0},  // -Deltax1, +Deltax1 [5, 6]
    {0, 0, 0, -1}, {0, 0, 0, 1},  // -Deltax0, +Deltax0 [7, 8]
    {-1, -1, 0, 0}, {1, -1, 0, 0}, {1, 1, 0, 0}, {-1, 1, 0, 0},  // ±Deltax3 ±Deltax2 [9–12]
    {-1, 0, -1, 0}, {1, 0, -1, 0}, {1, 0, 1, 0}, {-1, 0, 1, 0},  // ±Deltax3 ±Deltax1 [13–16]
    {0, -1, -1, 0}, {0, 1, -1, 0}, {0, 1, 1, 0}, {0, -1, 1, 0},  // ±Deltax2 ±Deltax1 [17–20]
    {-1, 0, 0, -1}, {1, 0, 0, -1}, {1, 0, 0, 1}, {-1, 0, 0, 1},  // ±Deltax3 ±Deltax0 [21–24]
    {0, -1, 0, -1}, {0, 1, 0, -1}, {0, 1, 0, 1}, {0, -1, 0, 1},  // ±Deltax2 ±Deltax0 [25–28]
    {0, 0, -1, -1}, {0, 0, 1, -1}, {0, 0, 1, 1}, {0, 0, -1, 1}   // ±Deltax1 ±Deltax0 [29–32]
    };
  for (entry = 0; entry < NSTENCIL; entry++)
    HYPRE_SStructStencilSetEntry(*stencil, entry, offsets[entry],var);
}

void model_set_stencil_values(HYPRE_SStructMatrix* A, long long int* ilower, long long int* iupper,
			      int ni, int nj, int nk, int nl, int pi, int pj, int pk, int pl,
			      double dx0, double dx1, double dx2, double dx3, int part, int var)
{
  int i, j;
  int nentries = NSTENCIL;
  int nvalues = nentries * ni * nj * nk * nl;
  double *values;
  long long int stencil_indices[NSTENCIL];
  
  values = (double*) calloc(nvalues, sizeof(double));
  
  for (j = 0; j < nentries; j++)
    stencil_indices[j] = j;
  
  for (i = 0; i < nvalues; i += nentries) {
    double x0, x1, x2,x3;
    double coeff[15];
    int gridi, gridj, gridk, gridl, temp;
    
    temp = i / nentries;

    gridk = temp / (ni * nj * nl); 
    gridl = (temp - gridk * ni * nj * nl) / (nj * ni);  
    gridj = (temp - gridk * ni * nj * nl - gridl * nj * ni) / ni ; 
    gridi = temp - gridk * ni * nj * nl - gridl * nj * ni - gridj * ni;

    gridl += pl * nl;
    gridj += pj * nj;
    gridk += pk * nk;
    gridi += pi * ni;

    x0 = param_x0start + dx0 * (gridk + 0.5);
    x1 = param_x1start + dx1 * (gridl + 0.5);
    x2 = param_x2start + dx2 * (gridj + 0.5);
    x3 = param_x3start + dx3 * (gridi + 0.5);
    
    param_coeff(coeff, x0, x1, x2, x3, dx0, dx1, dx2, dx3, 15);
    
    values[i]     = coeff[0];                      

    values[i+1]   = coeff[1];

    values[i+2]   = coeff[2];

    values[i+3]   = coeff[3];

    values[i+4]   = coeff[4];            

    values[i+5]   = coeff[5];

    values[i+6]   = coeff[6];         

    values[i+7]   = coeff[7];     

    values[i+8]   = coeff[8]; 

    values[i+9]   = - coeff[9];     

    values[i+10]  = coeff[9];  

    values[i+11]  = -coeff[9];   

    values[i+12]  = coeff[9];      

    values[i+13]  = -coeff[10];
                          
    values[i+14]  = coeff[10];   

    values[i+15]  = -coeff[10];   
               
    values[i+16]  = coeff[10];   

    values[i+17]  = -coeff[11];  

    values[i+18]  = coeff[11];                       

    values[i+19]  = -coeff[11];

    values[i+20]  = coeff[11];          

    values[i+21]  = -coeff[12];   

    values[i+22]  = coeff[12];       

    values[i+23]  = -coeff[12];   

    values[i+24]  = coeff[12];          

    values[i+25]  = -coeff[13];  

    values[i+26]  = coeff[13];   

    values[i+27]  = -coeff[13];  

    values[i+28]  = coeff[13];     

    values[i+29]  = -coeff[14];    

    values[i+30]  = coeff[14];   
                      
    values[i+31]  = -coeff[14]; 

    values[i+32]  = coeff[14]; 
  }
  
  HYPRE_SStructMatrixSetBoxValues(*A, part, ilower, iupper, var, nentries, stencil_indices, values);
  
  free(values);
}

void model_set_bound(HYPRE_SStructMatrix* A, int ni, int nj, int nk, int nl,
		     int pi, int pj, int pk,int pl, int npi, int npj, int npk, int npl,
		     double dx0, double dx1, double dx2, double dx3)
{
  (void) A;
  (void) ni;
  (void) nj;
  (void) nk;
  (void) nl;
  (void) pi;
  (void) pj;
  (void) pk;
  (void) pl;
  (void) npi;
  (void) npj;
  (void) npk;
  (void) npl;
  (void) dx0;
  (void) dx1;
  (void) dx2;
  (void) dx3;
}

double model_area(int i, int j, int k, int l, int ni, int nj, int nk,int nl,
		  int pi, int pj, int pk, int pl, double dx0, double dx1, double dx2, double dx3)
{
  (void) i;
  (void) j;
  (void) k;
  (void) l;
  (void) ni;
  (void) nj;
  (void) nk;
  (void) nl;
  (void) pi;
  (void) pj;
  (void) pk;
  (void) pl;
  (void) dx0;
  return dx1 * dx2 * dx3;
}
