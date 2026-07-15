#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "_hypre_utilities.h"
#include "HYPRE_sstruct_ls.h"
#include "HYPRE_parcsr_ls.h"
#include "hdf5_utils.h"
#include "param.h"
#include "model.h"

static void solve_once(HYPRE_SStructMatrix A, HYPRE_SStructVector b,
                       HYPRE_SStructVector x, int solver_id, int myid)
{
  HYPRE_ParCSRMatrix parA;
  HYPRE_ParVector parb;
  HYPRE_ParVector parx;
  HYPRE_Solver solver;
  HYPRE_Solver precond;

  HYPRE_SStructMatrixGetObject(A, (void **) &parA);
  HYPRE_SStructVectorGetObject(b, (void **) &parb);
  HYPRE_SStructVectorGetObject(x, (void **) &parx);

  HYPRE_BoomerAMGCreate(&precond);
  HYPRE_BoomerAMGSetMaxIter(precond, 1);
  HYPRE_BoomerAMGSetTol(precond, 0.0);
  HYPRE_BoomerAMGSetPrintLevel(precond, myid == 0 ? 1 : 0);

  if (solver_id == 1) {
    /* PCG is kept as an option for constant-coefficient or explicitly
       symmetrized runs.  The default is GMRES because the expanded
       variable-coefficient stencil is generally nonsymmetric. */
    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);
    HYPRE_ParCSRPCGSetTol(solver, 1.0e-04);
    HYPRE_ParCSRPCGSetPrintLevel(solver, myid == 0 ? 2 : 0);
    HYPRE_ParCSRPCGSetMaxIter(solver, 50);
    HYPRE_ParCSRPCGSetPrecond(solver, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, precond);
    HYPRE_ParCSRPCGSetup(solver, parA, parb, parx);
    HYPRE_ParCSRPCGSolve(solver, parA, parb, parx);
    HYPRE_ParCSRPCGDestroy(solver);
  } else {
    HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &solver);
    HYPRE_ParCSRGMRESSetTol(solver, 1.0e-04);
    HYPRE_ParCSRGMRESSetPrintLevel(solver, myid == 0 ? 2 : 0);
    HYPRE_ParCSRGMRESSetMaxIter(solver, 100);
    HYPRE_ParCSRGMRESSetKDim(solver, 30);
    HYPRE_ParCSRGMRESSetPrecond(solver, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, precond);
    HYPRE_ParCSRGMRESSetup(solver, parA, parb, parx);
    HYPRE_ParCSRGMRESSolve(solver, parA, parb, parx);
    HYPRE_ParCSRGMRESDestroy(solver);
  }

  HYPRE_BoomerAMGDestroy(precond);
}

int main (int argc, char *argv[])
{
  int i, j, k, l;
  
  int myid, num_procs;
  
  int ni, nj, nk, nl, pi, pj, pk, pl, npi, npj, npk, npl;
  double dx0, dx1, dx2, dx3;
  long long int ilower[4], iupper[4];
  
  int solver_id;
  int n_pre, n_post;
  
  clock_t start_t = clock();
  clock_t check_t;
  
  HYPRE_SStructGrid    grid;
  HYPRE_SStructGraph   graph;
  HYPRE_SStructStencil stencil;

  HYPRE_SStructMatrix  A;
  HYPRE_SStructVector  b;
  HYPRE_SStructVector  x;

  /* We only have one part and one variable */
  int nparts = 1;
  int nvars  = 1;
  int part   = 0;
  int var    = 0;

  /******************/
  /*No Preconditioner*/
  // int object_type = HYPRE_SSTRUCT;

  /*For BoomerAMG*/
  int object_type = HYPRE_PARCSR;

  /******************/
  
  int num_recursions; // number of recursions
  int nrecur_set;
  
  int output, timer, dump;

  char* dir_ptr;
  char* params_ptr;
  char* source_ptr;

  char filename[1024];
  
  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  /* Initialize HYPRE */
  HYPRE_Initialize();
  
  /* Set defaults */
  ni  = 32;                    
  npi = 1;
  npj = 1;
  npl = 1;
  npk = num_procs;             /* default processor grid is 1 x 1 x N */
  solver_id = 0;                  /* 0 = GMRES, 1 = PCG */
  n_pre  = 1;
  n_post = 1;
  output = 1;                  /* output data by default */
  timer  = 0;
  dump   = 0;                  /* outputing intermediate steps if nrecur > 1
				  off by default */
  num_recursions = 2;              /* L^2 F = source by default */
  nrecur_set = 0;
  char* default_dir = ".";     /* output in current directory by default */
  dir_ptr   = default_dir;
  params_ptr = NULL;
  source_ptr = NULL;
  
  /* Initiialize rng */
  const gsl_rng_type *T;
  gsl_rng *rstate;
  int mySeed= 1;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  rstate = gsl_rng_alloc(T);
  gsl_rng_set(rstate, model_set_gsl_seed(mySeed, myid));
  
  /* Parse command line */
  {
    int arg_index   = 0;
    int print_usage = 0;
    int check_pgrid = 0;
    
    while (arg_index < argc) 
      {
      if ( strcmp(argv[arg_index], "-n") == 0 ) 
        {
	        arg_index++;
	        ni = atoi(argv[arg_index++]);
        }
      else 
        {
	        arg_index++;
        }
       }
    nj = ni;
    nk = ni;
    nl = ni;
    
    arg_index = 0;
    while (arg_index < argc) 
      {
        if ( strcmp(argv[arg_index], "-ni") == 0 ) 
        {
	        arg_index++;
	        ni = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-nj") == 0 ) 
        {
	        arg_index++;
	        nj = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-nl") == 0 ) 
        {
	        arg_index++;
	        nl = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-nk") == 0 ) 
        {
          arg_index++;
          nk = atoi(argv[arg_index++]);
        }
      else if ( strcmp(argv[arg_index], "-pgrid") == 0 ) 
        {
	        arg_index++;
	        /* Make sure there are 4 arguments after -pgrid */
	        if (arg_index >= argc - 3) 
          {
	          check_pgrid = 1;
	          break;
	          }
	        /* Check that pgrid agrees with assigned number of processors
	        (Also checks for non-integer inputs) */
          npl = atoi(argv[arg_index++]);
          npj = atoi(argv[arg_index++]);
          npi = atoi(argv[arg_index++]);
          npk = atoi(argv[arg_index++]);
	        if ( num_procs != (npi * npj * npk * npl) ) 
          {
	          check_pgrid = 1;
	          break;
	        }
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 ) 
      {
	      arg_index++;
	      solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-v") == 0 ) 
      {
	      arg_index++;
	      n_pre = atoi(argv[arg_index++]);
	      n_post = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dryrun") == 0 ) 
      {
	      arg_index++;
	      output = 0;
      }
      else if ( strcmp(argv[arg_index], "-ps") == 0 || strcmp(argv[arg_index], "-sp") == 0 ) 
      {
	      arg_index++;
	      params_ptr = argv[arg_index];
	      source_ptr = argv[arg_index++];
      }
      else if ( strcmp(argv[arg_index], "-p") == 0 || strcmp(argv[arg_index], "-params") == 0 ) 
      {
	      arg_index++;
	      params_ptr = argv[arg_index++];
      }
      else if ( strcmp(argv[arg_index], "-s") == 0 || strcmp(argv[arg_index], "-source") == 0 ) 
      {
	      arg_index++;
	      source_ptr = argv[arg_index++];
      }
      else if ( strcmp(argv[arg_index], "-o") == 0 || strcmp(argv[arg_index], "-output") == 0 ) 
      {
	      arg_index++;
	      dir_ptr = argv[arg_index++];
      }
      else if ( strcmp(argv[arg_index], "-timer") == 0 ) 
      {
	      arg_index++;
	      timer = 1;
      }
      else if ( strcmp(argv[arg_index], "-dump") == 0 ) 
      {
	      arg_index++;
	      dump = 1;
      }
      else if ( strcmp(argv[arg_index], "-nrecur") == 0 ) 
      {
	      arg_index++;
	      num_recursions = atoi(argv[arg_index++]);
          nrecur_set = 1;
	      if (num_recursions < 1) 
        {
	        print_usage = 1;
	        break;
	      }
      }
      else if ( strcmp(argv[arg_index], "help")   == 0 || strcmp(argv[arg_index], "-help")  == 0 || strcmp(argv[arg_index], "--help") == 0 ) 
      {
	      print_usage = 1;
	      break;
      }
      else 
      {
	      arg_index++;
      }
    }
    
    if (print_usage) 
    {
      if (myid == 0) 
      {
        printf("\n");
        printf("Usage: %s [<options>]\n", argv[0]);
        printf("\n");
        printf("  -n <n>                      : General grid side length per processor (default: 32).\n");
        printf("  -ni <n> (or nj/nk)          : Grid side length for specific side per processor\n");
        printf("                                (default: 32). If both -n and -ni are specified, ni\n");
        printf("                                will overwrite n for that side only. \n");
        printf("  -pgrid <pl> <pj> <pi> <pk>  : Processor grid layout (default: 1 x 1 x num_procs).\n");
        printf("                                The product npi * npj * npl * npk must equal num_proc.\n");
        printf("  -solver <ID>                : solver ID\n");
        printf("                                 0  - GMRES with BoomerAMG preconditioner (default)\n");
        printf("                                 1  - PCG with BoomerAMG preconditioner\n");
        printf("  -v <n_pre> <n_post>         : Number of pre and post relaxations (default: 1 1).\n");
        printf("  -dryrun                     : Run solver w/o data output.\n");
        printf("  -params <file> (or -p)      : Read in parameters from <file>.\n");
        printf("  -source <file> (or -s)      : Read in source field from <file.\n");
        printf("                                Can be combined by using -ps or -sp\n");
        printf("  -output <dir> (or -o)       : Output data in directory <dir> (default: ./).\n");
        printf("  -timer                      : Time each step on processor zero.\n");
        printf("  -dump                       : Output intermediate steps if nrecur > 1).\n");
        printf("  -nrecur                     : Number of recursive L solves (default: 2 for nu=2).\n");
        printf("\n");
        printf("Sample run:     mpirun -np 8 poisson -n 32 -nk 64 -pgrid 1 2 4 2 -solver 1\n");
        printf("                mpiexec -n 4 ./disk -n 128 -nj 32 -pgrid 2 2 1 2 -solver 0\n");
        printf("\n");
        printf("GSL_RNG_SEED=${RANDOM} mpiexec -n 4 ./inoisy4d -n 64 -nk 128 -pgrid 1 1 1 4\n");
        printf("-solver 0 -timer -nrecur 2 -o output/\n");
        printf("\n");
      }
      MPI_Finalize();
      return (0);
    }
    
    if (check_pgrid) 
    {
      if (myid ==0) 
        {
          printf("Val: %i \n",num_procs);
	        printf("Error: Processor grid does not match the total number of processors. \n");
	        printf("       npi * npj * npl * npk must equal num_proc (see -help). \n");
        }
      MPI_Finalize();
      return (0);
    }	
  }

  if ((source_ptr != NULL) && (!nrecur_set))
    num_recursions = 1;

  (void) n_pre;
  (void) n_post;
  (void) dump;

  /* Read model parameters after command-line parsing and before matrix assembly. */
  param_read_params(params_ptr);

  /* Set dx0, dx1, dx2 */
  model_set_spacing(&dx0, &dx1, &dx2, &dx3, ni, nj, nk, nl, npi, npj, npk, npl);

  /* Figure out processor grid (npi x npj x npl x npk). Processor position 
     indicated by pi, pj, pk. Size of local grid on processor is 
     (ni x nj x nl x nk) */
  
  {
    int rank = myid;
    pi = rank % npi;
    rank /= npi;
    pj = rank % npj;
    rank /= npj;
    pl = rank % npl;
    rank /= npl;
    pk = rank;
  }
  
  /* Figure out the extents of each processor's piece of the grid. */
  ilower[0] = pi * ni; //z
  ilower[1] = pj * nj; //y
  ilower[2] = pl * nl; //x
  ilower[3] = pk * nk; //t
  
  iupper[0] = ilower[0] + ni - 1;
  iupper[1] = ilower[1] + nj - 1;
  iupper[2] = ilower[2] + nl - 1;
  iupper[3] = ilower[3] + nk - 1;

  //printf("Val 1: %lli \n",ilower[3]);
  //printf("Val 2: %lli \n",iupper[3]);

  /* Set up and assemble grid */
  {
    HYPRE_SStructGridCreate(MPI_COMM_WORLD, 4, nparts, &grid);

    HYPRE_SStructGridSetExtents(grid, part, ilower, iupper);

    /* Set the variable type and number of variables on each part. */
    {
       //HYPRE_SStructVariable vartypes[1] = {HYPRE_SSTRUCT_VARIABLE_NODE};
       HYPRE_SStructVariable vartypes[1] = {HYPRE_SSTRUCT_VARIABLE_CELL};

       HYPRE_SStructGridSetVariables(grid, part, nvars, vartypes);
    }

    printf("SStruct Grid Set \n");
    
    /* Set periodic boundary conditions of model */
    long long int bound_con[4];
    model_set_periodic(bound_con, ni, nj, nk, nl, npi, npj, npk, npl, 4);
    HYPRE_SStructGridSetPeriodic(grid, part, bound_con);
    
    HYPRE_SStructGridAssemble(grid);
  }
  
  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Grid initialized: t = %lf\n\n", (double)(check_t - start_t) / CLOCKS_PER_SEC);

  /* Initialize stencil and Struct Matrix, and set stencil values */
  model_create_stencil(&stencil, 4, var);

   /* 3. Set up the Graph - this determines the non-zero structure of the matrix
      and allows non-stencil relationships between the parts */
   {
      /* Create the graph object */
      HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &graph);

      /* See MatrixSetObjectType below */
      HYPRE_SStructGraphSetObjectType(graph, object_type);

      /* Now we need to tell the graph which stencil to use for each variable on
         each part (we only have one variable and one part) */
      HYPRE_SStructGraphSetStencil(graph, part, var, stencil);

      /* Here we could establish connections between parts if we had more than
         one part using the graph. For example, we could use
         HYPRE_GraphAddEntries() routine or HYPRE_GridSetNeighborPart() */

      /* Assemble the graph */
      HYPRE_SStructGraphAssemble(graph);
   }

  HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &A);
  HYPRE_SStructMatrixSetObjectType(A, object_type);
  HYPRE_SStructMatrixInitialize(A);

  model_set_stencil_values(&A, ilower, iupper, ni, nj, nk, nl, pi, pj, pk, pl, dx0, dx1, dx2, dx3, part, var);
  
  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Stencils values set: t = %lf\n\n",
	   (double)(check_t - start_t) / CLOCKS_PER_SEC);
	
  /* Fix boundary conditions and assemble Struct Matrix */
  model_set_bound(&A, ni, nj, nk, nl, pi, pj, pk, pl, npi, npj, npk, npl, dx0, dx1, dx2, dx3);
  
  HYPRE_SStructMatrixAssemble(A);

  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Boundary conditions set: t = %lf\n\n", (double)(check_t - start_t) / CLOCKS_PER_SEC);

  int    nvalues = ni * nj * nk * nl;
  double *values;
		
  values = (double*) calloc(nvalues, sizeof(double));
	
  /* Set up SStruct Vectors for b and x */
  {
  	
    HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &b);
    HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &x);

    /* As with the matrix, set the appropriate object type for the vectors */
    HYPRE_SStructVectorSetObjectType(b, object_type);
    HYPRE_SStructVectorSetObjectType(x, object_type);
		
    HYPRE_SStructVectorInitialize(b);
    HYPRE_SStructVectorInitialize(x);

    if ( source_ptr == NULL )
    {
      param_set_source(values, rstate, ni, nj, nk, nl, pi, pj, pk, pl, npi, npj, npk, npl, dx0, dx1, dx2, dx3, num_recursions);
    }
    else
    {
      if (myid ==0) 
        {
          printf("Reading a source file \n\n");
        }
      hdf5_open(source_ptr);
      
      hdf5_set_directory("/data/");

      hsize_t fdims[4]  = {npk * nk, npl * nl, npj * nj, npi * ni};
      hsize_t fstart[4] = {pk * nk, pl * nl, pj * nj, pi * ni};
      hsize_t fcount[4] = {nk, nl, nj, ni};
      hsize_t mdims[4]  = {nk, nl, nj, ni};
      hsize_t mstart[4] = {0, 0, 0, 0};
      
      hdf5_read_array(values, "data_raw", 4, fdims, fstart, fcount, mdims, mstart, H5T_NATIVE_DOUBLE);

      hdf5_close();

    }
		

    HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, values);
		
    for (i = 0; i < nvalues; i ++)
      {
        values[i] = 0.0;
      }
      
    HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, values);
		
    free(values);
		
    HYPRE_SStructVectorAssemble(b);
    HYPRE_SStructVectorAssemble(x);
  }

  check_t = clock();
  if ( (myid == 0) && (timer) )
    {
      printf("SStruct vector assembled: t = %lf\n\n", (double)(check_t - start_t) / CLOCKS_PER_SEC);
    }

  /* Setup output file */
  
  if (myid == 0)
    param_set_output_name(filename, sizeof(filename), ni, nj, nk, nl, npi, npj, npk, npl, dir_ptr);

  MPI_Bcast(filename, sizeof(filename), MPI_CHAR, 0, MPI_COMM_WORLD);  
  
  if (output) {
    hdf5_create(filename);
    hdf5_set_directory("/");
    hdf5_make_directory("data");
    hdf5_set_directory("/data/");
  }
  
  /**********************/

  for (int recur = 0; recur < num_recursions; recur++) {
    if (myid == 0)
      printf("SPDE solve stage %d of %d\n", recur + 1, num_recursions);

    solve_once(A, b, x, solver_id, myid);

    if (recur < num_recursions - 1) {
      double *stage_values = (double*) calloc(nvalues, sizeof(double));
      HYPRE_SStructVectorGather(x);
      HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper, var, stage_values);

      HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper, var, stage_values);

      for (i = 0; i < nvalues; i++)
        stage_values[i] = 0.0;
      HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper, var, stage_values);

      HYPRE_SStructVectorAssemble(b);
      HYPRE_SStructVectorAssemble(x);
      free(stage_values);
    }
  }

  /**********************/
  
  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Solver finished: t = %lf\n\n", (double)(check_t - start_t) / CLOCKS_PER_SEC);	
  
  /* Output data */
  if (output) {
    /* Get the local raw data */
    int nvalues = ni * nj * nk * nl;
    
    double *raw = (double*)calloc(nvalues, sizeof(double));

    //Added to test
    HYPRE_SStructVectorGather(x);
    HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper, var, raw);  
  
    /* Find statistics for raw data and envelope */
    double avg_raw = 0.;
    double var_raw = 0.;
    double min_raw = raw[0];
    double max_raw = raw[0];
    
    double *lc_raw = (double*)calloc(npk * nk, sizeof(double));

    /* Calculate mean and variance for raw data */

    {
      for(i = 0; i < nvalues; i++)
	   avg_raw += raw[i];
      
      MPI_Allreduce(MPI_IN_PLACE, &avg_raw, 1, MPI_DOUBLE,
		    MPI_SUM, MPI_COMM_WORLD);
      
      avg_raw /= (npi * npj * npk * npl * nvalues);
      
      /* Second pass for variance */
      for (i = 0; i < nvalues; i++)
	        var_raw += (raw[i] - avg_raw) * (raw[i] - avg_raw);
      
      MPI_Allreduce(MPI_IN_PLACE, &var_raw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      
      var_raw /= (npi * npj * npk * npl * nvalues - 1);
    }

    /* Find min, max, lightcurve*/
    {
      int m = 0;
      double area;
      for (k = pk * nk; k < (pk + 1) * nk; k++) {
	for (l = 0; l < nl; l++) {
	  for (j = 0; j < nj; j++) {
      for (i = 0; i < ni; i++) {
	    area = model_area(i, j, k, l, ni, nj, nk, nl, pi, pj, pk, pl, dx0, dx1, dx2, dx3);

	    lc_raw[k] += raw[m] * area;
	    if (raw[m] < min_raw)
	      min_raw = raw[m];
	    if (raw[m] > max_raw)
	      max_raw = raw[m];
	    m++;
	  }
	}
      }
    }
  }

    // (currently required for hdf5_write_single_val)
    MPI_Allreduce(MPI_IN_PLACE, &min_raw, 1, MPI_DOUBLE,MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_raw, 1, MPI_DOUBLE,MPI_MAX, MPI_COMM_WORLD);

    if (myid == 0) {
      MPI_Reduce(MPI_IN_PLACE, lc_raw, npk * nk, MPI_DOUBLE,MPI_SUM, 0, MPI_COMM_WORLD);
    }
    else 
    {
      MPI_Reduce(lc_raw, lc_raw, npk * nk, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    /* File i/o */

    /* Save solution to output file*/
    hdf5_set_directory("/");
    hdf5_set_directory("/data/");
    /* note: HYPRE has k as the slowest varying, opposite of HDF5 */
    {
      hsize_t fdims[4]  = {npk * nk, npl * nl, npj * nj, npi * ni};
      hsize_t fstart[4] = {pk * nk, pl * nl, pj * nj, pi * ni};
      hsize_t fcount[4] = {nk, nl, nj, ni};
      hsize_t mdims[4]  = {nk, nl, nj, ni};
      hsize_t mstart[4] = {0, 0, 0, 0};

      hdf5_write_array(raw, "data_raw", 4, fdims, fstart, fcount,mdims, mstart, H5T_NATIVE_DOUBLE);  
    }
    
    /* Output lightcurve and parameters */
    {
      hsize_t fdims  = npk * nk;
      hsize_t fstart = 0;
      hsize_t fcount = 0;
      hsize_t mdims  = 0;
      hsize_t mstart = 0;

      if (myid == 0) {
	      fcount = npk * nk;
	      mdims  = npk * nk;
      }
	
      hdf5_write_array(lc_raw, "lc_raw", 1, &fdims, &fstart, &fcount, &mdims, &mstart, H5T_NATIVE_DOUBLE);
    }
    
    hdf5_set_directory("/");
    hdf5_make_directory("params");
    hdf5_set_directory("/params/");

    hdf5_write_single_val(&param_x0start, "x0start", H5T_IEEE_F64LE);
    hdf5_write_single_val(&param_x0end, "x0end", H5T_IEEE_F64LE);
    hdf5_write_single_val(&param_x1start, "x1start", H5T_IEEE_F64LE);
    hdf5_write_single_val(&param_x1end, "x1end", H5T_IEEE_F64LE);
    hdf5_write_single_val(&param_x2start, "x2start", H5T_IEEE_F64LE);
    hdf5_write_single_val(&param_x2end, "x2end", H5T_IEEE_F64LE);
    hdf5_write_single_val(&param_x3start, "x3start", H5T_IEEE_F64LE);
    hdf5_write_single_val(&param_x3end, "x3end", H5T_IEEE_F64LE);
    hdf5_write_single_val(&dx0, "dx0", H5T_IEEE_F64LE);
    hdf5_write_single_val(&dx1, "dx1", H5T_IEEE_F64LE);
    hdf5_write_single_val(&dx2, "dx2", H5T_IEEE_F64LE);
    hdf5_write_single_val(&dx3, "dx3", H5T_IEEE_F64LE);

    hdf5_write_single_val(&npi, "npi", H5T_STD_I32LE);
    hdf5_write_single_val(&npj, "npj", H5T_STD_I32LE);
    hdf5_write_single_val(&npk, "npk", H5T_STD_I32LE);
    hdf5_write_single_val(&npl, "npl", H5T_STD_I32LE);
    hdf5_write_single_val(&ni, "ni", H5T_STD_I32LE);
    hdf5_write_single_val(&nj, "nj", H5T_STD_I32LE);
    hdf5_write_single_val(&nk, "nk", H5T_STD_I32LE);
    hdf5_write_single_val(&nl, "nl", H5T_STD_I32LE);
    hdf5_write_single_val(&gsl_rng_default_seed, "seed", H5T_STD_U64LE);
    hdf5_write_single_val(&min_raw, "min_raw", H5T_IEEE_F64LE);
    hdf5_write_single_val(&max_raw, "max_raw", H5T_IEEE_F64LE);
    hdf5_write_single_val(&avg_raw, "avg_raw", H5T_IEEE_F64LE);
    hdf5_write_single_val(&var_raw, "var_raw", H5T_IEEE_F64LE);
    param_write_params(filename);
    hdf5_close();

    check_t = clock();
    if ( (myid == 0) && (timer) )
      printf("Data output: t = %lf\n\n",
	     (double)(check_t - start_t) / CLOCKS_PER_SEC);

    if (myid == 0)
      printf("%s\n\n", filename);
    
    free(raw);
    free(lc_raw);
  }
  
 /* Free memory */
 HYPRE_SStructGridDestroy(grid);
 HYPRE_SStructStencilDestroy(stencil);
 HYPRE_SStructGraphDestroy(graph);
 HYPRE_SStructMatrixDestroy(A);
 HYPRE_SStructVectorDestroy(b);
 HYPRE_SStructVectorDestroy(x);

  /* Free GSL rng state */
  gsl_rng_free(rstate);

  /* Finalize HYPRE */
  HYPRE_Finalize();
  
  /* Finalize MPI */
  MPI_Finalize();
  
  return (0);
}
