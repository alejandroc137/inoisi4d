#include "param.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "hdf5_utils.h"

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

#define SMALL 1.e-12
#define OMEGA_FLOOR 1.e-8
#define LAMBDA_FLOOR 1.e-8

const char model_name[] = "inoisyPlus";

/* Grid domain */
const double param_x0start = 0.;         /* t in terms of M */
const double param_x0end   = 256.;
const double param_x1start = -30.;       /* x in terms of M */
const double param_x1end   = 30.;
const double param_x2start = -30.;       /* y in terms of M */
const double param_x2end   = 30.;
const double param_x3start = -50.;       /* z in terms of M */
const double param_x3end   = 50.;

/* Component selector for the single scalar GRF.
   0 = disk/torus only, 1 = jet only, 2 = composite torus+jet tensor.
   In mode 2 the code still solves one scalar SPDE; the disk and jet enter only
   through a spatially varying covariance tensor. */
static int param_component = 2;

/* Disk/torus correlation lengths.  If lam0 <= 0, the disk correlation time is
   set locally to 2 pi / |Omega|. */
static double param_lam0 = -1.0;
static double param_lam1 = 5.0;
static double param_lam2 = 1.5;
static double param_lam3 = 0.6;

/* Jet correlation lengths. */
static double param_jet_lam0 = 10.0;
static double param_jet_lam1 = 2.5;
static double param_jet_lam2 = 2.5;
static double param_jet_lam3 = 2.5;

/* Disk pitch angle p and jet helical angle chi. */
static double param_pitch = M_PI / 20.0;
static double param_jet_chi0 = M_PI / 20.0;
static double param_jet_chi_rho = 2.0;

/* Torus geometry used only by the GRF correlations, not by the four-velocity.
   The disk weight is exp[-0.5*((rho-r0)/ar)^2 -0.5*(z/az)^2]. */
static double torus_r0 = 12.0;
static double torus_ar = 8.0;
static double torus_az = 5.0;

/* Jet geometry used only by the GRF correlations.  The jet transverse width is
   jet_width + jet_opening*|z|. */
static double jet_width = 2.5;
static double jet_opening = 0.30;

/* Composite-tensor controls.  If normalize_weights=0, the raw GRF is naturally
   concentrated in the torus and jet.  If normalize_weights=1, the weights only
   interpolate the orientation/correlation tensor and do not localize the field. */
static double blend_floor = 1.e-6;
static int normalize_weights = 0;

/* Kerr four-velocity parameters. */
static double spin = 0.94;
static double sub_kep = 1.0;       /* xi */
static double ell_delta = 3.0;     /* delta */
static int orbit_sigma = 1;        /* +1 prograde, -1 retrograde */
static double betar = 0.8;
static int use_plunge = 0;
static int enforce_timelike = 1;

/* Simple jet advection model: v = jet_vpol e_r + rho jet_omega e_phi. */
static double jet_vpol = 0.5;
static double jet_omega = 0.0;

static double clamp(double x, double lo, double hi)
{
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

static double positive(double x)
{
  if (!isfinite(x)) return LAMBDA_FLOOR;
  return fmax(fabs(x), LAMBDA_FLOOR);
}

static void read_double_if_exists(const char *name, double *val)
{
  if (hdf5_exists(name)) hdf5_read_single_val(val, name, H5T_IEEE_F64LE);
}

static void read_int_if_exists(const char *name, int *val)
{
  if (hdf5_exists(name)) hdf5_read_single_val(val, name, H5T_STD_I32LE);
}

/* Read in parameters from input file.  Missing datasets are ignored so old
   parameter files remain usable. */
void param_read_params(char* filename)
{
  if (filename != NULL) {
    hdf5_open(filename);
    hdf5_set_directory("/params/");

    read_int_if_exists("component", &param_component);

    read_double_if_exists("lam0", &param_lam0);
    read_double_if_exists("lam1", &param_lam1);
    read_double_if_exists("lam2", &param_lam2);
    read_double_if_exists("lam3", &param_lam3);

    read_double_if_exists("jet_lam0", &param_jet_lam0);
    read_double_if_exists("jet_lam1", &param_jet_lam1);
    read_double_if_exists("jet_lam2", &param_jet_lam2);
    read_double_if_exists("jet_lam3", &param_jet_lam3);

    read_double_if_exists("pitch", &param_pitch);
    read_double_if_exists("jet_chi0", &param_jet_chi0);
    read_double_if_exists("jet_chi_rho", &param_jet_chi_rho);

    read_double_if_exists("torus_r0", &torus_r0);
    read_double_if_exists("torus_ar", &torus_ar);
    read_double_if_exists("torus_az", &torus_az);
    read_double_if_exists("jet_width", &jet_width);
    read_double_if_exists("jet_opening", &jet_opening);
    read_double_if_exists("blend_floor", &blend_floor);
    read_int_if_exists("normalize_weights", &normalize_weights);

    read_double_if_exists("spin", &spin);
    read_double_if_exists("xi", &sub_kep);
    read_double_if_exists("delta", &ell_delta);
    read_int_if_exists("sigma", &orbit_sigma);
    read_double_if_exists("betar", &betar);
    read_int_if_exists("use_plunge", &use_plunge);
    read_int_if_exists("enforce_timelike", &enforce_timelike);

    read_double_if_exists("jet_vpol", &jet_vpol);
    read_double_if_exists("jet_omega", &jet_omega);

    hdf5_close();
  }

  if (orbit_sigma >= 0) orbit_sigma = 1;
  else orbit_sigma = -1;
  if (param_component < 0 || param_component > 2) param_component = 2;
  spin = clamp(spin, -0.999999, 0.999999);
  betar = clamp(betar, 0.0, 1.0);
  torus_r0 = positive(torus_r0);
  torus_ar = positive(torus_ar);
  torus_az = positive(torus_az);
  jet_width = positive(jet_width);
  jet_opening = fmax(0.0, jet_opening);
  blend_floor = fmax(0.0, blend_floor);
  normalize_weights = normalize_weights ? 1 : 0;
}

/* Write out parameters to output file. */
void param_write_params(char* filename)
{
  (void) filename;

  hdf5_set_directory("/params/");

  hdf5_write_single_val(&param_component, "component", H5T_STD_I32LE);

  hdf5_write_single_val(&param_lam0, "lam0", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_lam1, "lam1", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_lam2, "lam2", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_lam3, "lam3", H5T_IEEE_F64LE);

  hdf5_write_single_val(&param_jet_lam0, "jet_lam0", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_jet_lam1, "jet_lam1", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_jet_lam2, "jet_lam2", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_jet_lam3, "jet_lam3", H5T_IEEE_F64LE);

  hdf5_write_single_val(&param_pitch, "pitch", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_jet_chi0, "jet_chi0", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_jet_chi_rho, "jet_chi_rho", H5T_IEEE_F64LE);

  hdf5_write_single_val(&torus_r0, "torus_r0", H5T_IEEE_F64LE);
  hdf5_write_single_val(&torus_ar, "torus_ar", H5T_IEEE_F64LE);
  hdf5_write_single_val(&torus_az, "torus_az", H5T_IEEE_F64LE);
  hdf5_write_single_val(&jet_width, "jet_width", H5T_IEEE_F64LE);
  hdf5_write_single_val(&jet_opening, "jet_opening", H5T_IEEE_F64LE);
  hdf5_write_single_val(&blend_floor, "blend_floor", H5T_IEEE_F64LE);
  hdf5_write_single_val(&normalize_weights, "normalize_weights", H5T_STD_I32LE);

  hdf5_write_single_val(&spin, "spin", H5T_IEEE_F64LE);
  hdf5_write_single_val(&sub_kep, "xi", H5T_IEEE_F64LE);
  hdf5_write_single_val(&ell_delta, "delta", H5T_IEEE_F64LE);
  hdf5_write_single_val(&orbit_sigma, "sigma", H5T_STD_I32LE);
  hdf5_write_single_val(&betar, "betar", H5T_IEEE_F64LE);
  hdf5_write_single_val(&use_plunge, "use_plunge", H5T_STD_I32LE);
  hdf5_write_single_val(&enforce_timelike, "enforce_timelike", H5T_STD_I32LE);

  hdf5_write_single_val(&jet_vpol, "jet_vpol", H5T_IEEE_F64LE);
  hdf5_write_single_val(&jet_omega, "jet_omega", H5T_IEEE_F64LE);
}

void param_set_output_name(char* filename, size_t filename_size,
                           int ni, int nj, int nk, int nl,
                           int npi, int npj, int npk, int npl, char* dir)
{
  time_t rawtime;
  struct tm *timeinfo;
  char buffer[255];
  const char *component = (param_component == 1) ? "jet" : ((param_component == 2) ? "torusjet" : "disk");

  (void) ni;
  (void) nl;
  (void) npi;
  (void) npl;

  time(&rawtime);
  timeinfo = localtime(&rawtime);
  strftime(buffer, 255, "%Y_%m_%d_%H%M%S", timeinfo);

  int n = snprintf(filename, filename_size,
                   "%s/%s_%s_%s_%d_%d_%.0f_%.0f_a%.4f_xi%.2f_delta%.2f_br%.2f_l%.2f_%.2f_%.2f.h5",
                   dir, model_name, component, buffer, npj * nj, npk * nk,
                   param_x1end, param_x0end, spin, sub_kep, ell_delta, betar,
                   param_lam1, param_lam2, param_lam3);
  if (n < 0 || (size_t)n >= filename_size) {
    fprintf(stderr, "Output filename is too long; use a shorter output directory or filename template.\n");
    exit(EXIT_FAILURE);
  }
}

typedef struct {
  double r;
  double rho;
  double sinth;
  double costh;
  double phi;
  double xnorm;
  double ynorm;
  double znorm;
} bl_point;

typedef struct {
  double delta;
  double sigma;
  double A;
  double gtt;
  double gtphi;
  double grr;
  double gphiphi;
  double gtt_inv;
  double gtphi_inv;
  double grr_inv;
  double gphiphi_inv;
} kerr_metric;

static double horizon_radius(double a)
{
  return 1.0 + sqrt(fmax(0.0, 1.0 - a*a));
}

static void set_bl_point(bl_point *p, double x, double y, double z)
{
  const double r_raw = sqrt(x*x + y*y + z*z);
  const double rho = sqrt(x*x + y*y);
  const double r_min = horizon_radius(spin) + 1.e-6;

  p->rho = rho;
  p->phi = atan2(y, x);

  if (r_raw > SMALL) {
    p->sinth = rho / r_raw;
    p->costh = z / r_raw;
    p->xnorm = x / r_raw;
    p->ynorm = y / r_raw;
    p->znorm = z / r_raw;
  } else {
    p->sinth = 1.0;
    p->costh = 0.0;
    p->xnorm = 0.0;
    p->ynorm = 0.0;
    p->znorm = 0.0;
  }

  p->r = fmax(r_raw, r_min);
}

static void set_kerr_metric(kerr_metric *m, double r, double sinth, double costh, double a)
{
  double sin2 = fmax(sinth*sinth, 1.e-14);
  double cos2 = costh*costh;
  double Sigma = r*r + a*a*cos2;

  m->delta = r*r - 2.0*r + a*a;
  m->delta = fmax(m->delta, 1.e-14);
  m->A = (r*r + a*a)*(r*r + a*a) - a*a*m->delta*sin2;
  m->sigma = Sigma;

  m->gtt = -(1.0 - 2.0*r/Sigma);
  m->gtphi = -2.0*a*r*sin2/Sigma;
  m->grr = Sigma/m->delta;
  m->gphiphi = m->A*sin2/Sigma;

  m->gtt_inv = -m->A/(Sigma*m->delta);
  m->gtphi_inv = -2.0*a*r/(Sigma*m->delta);
  m->grr_inv = m->delta/Sigma;
  m->gphiphi_inv = (m->delta - a*a*sin2)/(Sigma*m->delta*sin2);
}

static double r_isco(double a, int sigma)
{
  const double aa = fabs(a);
  const double z1 = 1.0 + pow(1.0 - aa*aa, 1.0/3.0)
                    * (pow(1.0 + aa, 1.0/3.0) + pow(1.0 - aa, 1.0/3.0));
  const double z2 = sqrt(3.0*aa*aa + z1*z1);
  const double branch = (sigma >= 0) ? -1.0 : 1.0;
  return 3.0 + z2 + branch * sqrt((3.0 - z1)*(3.0 + z1 + 2.0*z2));
}

static double ell_profile(double rho)
{
  const double r = fmax(rho, 1.e-6);
  const double sr = sqrt(r);
  const double sigma = (orbit_sigma >= 0) ? 1.0 : -1.0;
  const double a = spin;
  double denom = r - 2.0 + ell_delta + sigma*a/sr;

  if (fabs(denom) < 1.e-10)
    denom = (denom >= 0.0) ? 1.e-10 : -1.e-10;

  return sub_kep * sigma * (r*sr - 2.0*sigma*a + a*a/sr) / denom;
}

static double U_of_ell(double ell, const kerr_metric *m)
{
  return -m->gtt_inv + 2.0*ell*m->gtphi_inv - ell*ell*m->gphiphi_inv;
}

static double omega_from_ell(double ell, const kerr_metric *m)
{
  const double denom = m->gtt_inv - ell*m->gtphi_inv;
  if (fabs(denom) < SMALL) return 0.0;
  return (m->gtphi_inv - ell*m->gphiphi_inv) / denom;
}

static double clamp_omega(double omega, const kerr_metric *m)
{
  if (!enforce_timelike || m->gphiphi <= SMALL) return omega;

  double disc = m->gtphi*m->gtphi - m->gtt*m->gphiphi;
  disc = sqrt(fmax(disc, 0.0));

  double om1 = (-m->gtphi - disc) / m->gphiphi;
  double om2 = (-m->gtphi + disc) / m->gphiphi;
  double omin = fmin(om1, om2);
  double omax = fmax(om1, om2);
  double eps = 1.e-8 * fmax(1.0, omax - omin);

  return clamp(omega, omin + eps, omax - eps);
}

static double D_of_omega(double omega, const kerr_metric *m)
{
  return -m->gtt - 2.0*m->gtphi*omega - m->gphiphi*omega*omega;
}

static double freefall_ur(const kerr_metric *m)
{
  return -sqrt(fmax(0.0, (-1.0 - m->gtt_inv) * m->grr_inv));
}

static double plunge_hat_ur(const bl_point *p, const kerr_metric *m)
{
  if (!use_plunge) return 0.0;

  const double rms = r_isco(spin, orbit_sigma);
  if (p->r >= rms) return 0.0;

  const double rb = rms;
  const double sinb = p->sinth;
  const double cosb = p->costh;
  const double rho_b = rb * sinb;
  const double ell_b = ell_profile(rho_b);

  kerr_metric mb;
  set_kerr_metric(&mb, rb, sinb, cosb, spin);

  const double Ub = U_of_ell(ell_b, &mb);
  if (Ub <= SMALL || !isfinite(Ub)) return 0.0;

  const double Eb = 1.0 / sqrt(Ub);
  const double Lb = Eb * ell_b;
  const double Rpl = -1.0 - m->gtt_inv*Eb*Eb
                     + 2.0*m->gtphi_inv*Eb*Lb
                     - m->gphiphi_inv*Lb*Lb;

  if (Rpl <= 0.0 || !isfinite(Rpl)) return 0.0;
  return -sqrt(fmax(0.0, m->grr_inv * Rpl));
}

static void disk_velocity(double v[3], double x0, double x1, double x2, double x3)
{
  (void) x0;

  bl_point p;
  kerr_metric m;
  set_bl_point(&p, x1, x2, x3);
  set_kerr_metric(&m, p.r, p.sinth, p.costh, spin);

  double ell = ell_profile(fmax(p.rho, 1.e-6));
  double omega = clamp_omega(omega_from_ell(ell, &m), &m);

  double hat_ur = plunge_hat_ur(&p, &m);
  double bar_ur = freefall_ur(&m);
  double ur = betar * hat_ur + (1.0 - betar) * bar_ur;
  double D = D_of_omega(omega, &m);

  if (D <= SMALL || !isfinite(D)) {
    omega = 0.0;
    ur = 0.0;
    D = D_of_omega(omega, &m);
  }

  const double ut = sqrt(fmax(SMALL, (1.0 + m.grr*ur*ur) / fmax(D, SMALL)));
  const double vr = ur / ut;

  v[0] = vr*p.xnorm - omega*x2;
  v[1] = vr*p.ynorm + omega*x1;
  v[2] = vr*p.znorm;
}

static double disk_omega(double x0, double x1, double x2, double x3)
{
  (void) x0;

  bl_point p;
  kerr_metric m;
  set_bl_point(&p, x1, x2, x3);
  set_kerr_metric(&m, p.r, p.sinth, p.costh, spin);

  return clamp_omega(omega_from_ell(ell_profile(fmax(p.rho, 1.e-6)), &m), &m);
}

static void jet_velocity(double v[3], double x0, double x1, double x2, double x3)
{
  (void) x0;

  const double r = sqrt(x1*x1 + x2*x2 + x3*x3);
  double er[3] = {0.0, 0.0, (x3 >= 0.0) ? 1.0 : -1.0};

  if (r > SMALL) {
    er[0] = x1 / r;
    er[1] = x2 / r;
    er[2] = x3 / r;
  }

  v[0] = jet_vpol*er[0] - jet_omega*x2;
  v[1] = jet_vpol*er[1] + jet_omega*x1;
  v[2] = jet_vpol*er[2];
}

static void set_q0_from_velocity(double q0[4], const double v[3])
{
  q0[0] = 1.0;
  q0[1] = v[0];
  q0[2] = v[1];
  q0[3] = v[2];
}

static void set_disk_advection_vector(double q0[4], double x0, double x1, double x2, double x3)
{
  double v[3];
  disk_velocity(v, x0, x1, x2, x3);
  set_q0_from_velocity(q0, v);
}

static void set_jet_advection_vector(double q0[4], double x0, double x1, double x2, double x3)
{
  double v[3];
  jet_velocity(v, x0, x1, x2, x3);
  set_q0_from_velocity(q0, v);
}

static void normalize3(double v[3], const double fallback[3])
{
  const double n = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  if (n > SMALL && isfinite(n)) {
    v[0] /= n;
    v[1] /= n;
    v[2] /= n;
  } else {
    v[0] = fallback[0];
    v[1] = fallback[1];
    v[2] = fallback[2];
  }
}

static void set_cylindrical_frame(double erho[3], double ephi[3], double x1, double x2)
{
  const double rho = sqrt(x1*x1 + x2*x2);

  erho[0] = 1.0;
  erho[1] = 0.0;
  erho[2] = 0.0;

  ephi[0] = 0.0;
  ephi[1] = 1.0;
  ephi[2] = 0.0;

  if (rho > SMALL) {
    erho[0] = x1 / rho;
    erho[1] = x2 / rho;
    erho[2] = 0.0;

    ephi[0] = -x2 / rho;
    ephi[1] = x1 / rho;
    ephi[2] = 0.0;
  }
}

static double torus_weight(double x1, double x2, double x3)
{
  const double rho = sqrt(x1*x1 + x2*x2);
  const double u = (rho - torus_r0) / fmax(torus_ar, LAMBDA_FLOOR);
  const double v = x3 / fmax(torus_az, LAMBDA_FLOOR);

  return exp(-0.5 * (u*u + v*v));
}

static double collimated_jet_weight(double x1, double x2, double x3)
{
  const double rho = sqrt(x1*x1 + x2*x2);
  const double width = fmax(jet_width + jet_opening*fabs(x3), LAMBDA_FLOOR);
  return exp(-0.5 * rho*rho / (width*width));
}

static void composite_weights(double *wd, double *wj, double x1, double x2, double x3)
{
  double d = torus_weight(x1, x2, x3);
  double j = collimated_jet_weight(x1, x2, x3);

  if (normalize_weights) {
    const double sum = d + j + 2.0*blend_floor;
    *wd = (d + blend_floor) / fmax(sum, SMALL);
    *wj = (j + blend_floor) / fmax(sum, SMALL);
  } else {
    *wd = d + blend_floor;
    *wj = j + blend_floor;
  }
}

static void set_disk_triad(double e1[4], double e2[4], double e3[4],
                           double x1, double x2, double x3)
{
  const double rho = sqrt(x1*x1 + x2*x2);
  const double u = (rho - torus_r0) / fmax(torus_ar, LAMBDA_FLOOR);
  const double v = x3 / fmax(torus_az, LAMBDA_FLOOR);

  double erho[3], ephi[3];
  double epol[3], enorm[3];
  const double fallback_pol[3] = {0.0, 0.0, 1.0};
  double fallback_norm[3] = {1.0, 0.0, 0.0};

  set_cylindrical_frame(erho, ephi, x1, x2);
  fallback_norm[0] = erho[0];
  fallback_norm[1] = erho[1];
  fallback_norm[2] = erho[2];

  /* Elliptical torus cross-section.  epol is tangent to the meridional
     torus cross-section; enorm is the corresponding cross-section normal. */
  epol[0] = -torus_ar * v * erho[0];
  epol[1] = -torus_ar * v * erho[1];
  epol[2] =  torus_az * u;

  enorm[0] =  torus_az * u * erho[0];
  enorm[1] =  torus_az * u * erho[1];
  enorm[2] =  torus_ar * v;

  normalize3(epol, fallback_pol);
  normalize3(enorm, fallback_norm);

  const double cp = cos(param_pitch);
  const double sp = sin(param_pitch);

  e1[0] = e2[0] = e3[0] = 0.0;

  /* Long disk correlations follow a spiral/helical direction on the torus
     surface rather than a purely cylindrical spiral at fixed z. */
  e1[1] = cp*ephi[0] + sp*epol[0];
  e1[2] = cp*ephi[1] + sp*epol[1];
  e1[3] = cp*ephi[2] + sp*epol[2];

  e2[1] = -sp*ephi[0] + cp*epol[0];
  e2[2] = -sp*ephi[1] + cp*epol[1];
  e2[3] = -sp*ephi[2] + cp*epol[2];

  e3[1] = enorm[0];
  e3[2] = enorm[1];
  e3[3] = enorm[2];
}

static void set_jet_triad(double e1[4], double e2[4], double e3[4],
                          double x1, double x2, double x3)
{
  const double rho = sqrt(x1*x1 + x2*x2);
  const double r = sqrt(rho*rho + x3*x3);

  double er[3] = {0.0, 0.0, (x3 >= 0.0) ? 1.0 : -1.0};
  double eth[3] = {1.0, 0.0, 0.0};
  double eph[3] = {0.0, 1.0, 0.0};

  if (rho > SMALL) {
    eph[0] = -x2 / rho;
    eph[1] = x1 / rho;
    eph[2] = 0.0;
  }

  if (r > SMALL && rho > SMALL) {
    er[0] = x1 / r;
    er[1] = x2 / r;
    er[2] = x3 / r;

    eth[0] = x1*x3/(rho*r);
    eth[1] = x2*x3/(rho*r);
    eth[2] = -rho/r;
  }

  const double rho_chi = fmax(param_jet_chi_rho, SMALL);
  const double chi = param_jet_chi0 * (1.0 - exp(-rho*rho/(rho_chi*rho_chi)));
  const double cc = cos(chi);
  const double sc = sin(chi);

  e1[0] = e2[0] = e3[0] = 0.0;

  e1[1] = cc*er[0] + sc*eph[0];
  e1[2] = cc*er[1] + sc*eph[1];
  e1[3] = cc*er[2] + sc*eph[2];

  e2[1] = eth[0];
  e2[2] = eth[1];
  e2[3] = eth[2];

  e3[1] = -sc*er[0] + cc*eph[0];
  e3[2] = -sc*er[1] + cc*eph[1];
  e3[3] = -sc*er[2] + cc*eph[2];
}

static double disk_corr_time(double x0, double x1, double x2, double x3)
{
  if (param_lam0 > 0.0)
    return positive(param_lam0);

  const double omega = fabs(disk_omega(x0, x1, x2, x3));
  const double lam_orb = 2.0 * M_PI / fmax(omega, OMEGA_FLOOR);

  const double lam_min = 1.0;
  const double lam_max = 1000.0;

  return clamp(lam_orb, lam_min, lam_max);
}

static void disk_corr_lengths(double lambda[4], double x0, double x1, double x2, double x3)
{
  (void) x1;
  (void) x2;

  lambda[0] = disk_corr_time(x0, x1, x2, x3);
  lambda[1] = positive(param_lam1);
  lambda[2] = positive(param_lam2);
  lambda[3] = positive(param_lam3);
}

static void jet_corr_lengths(double lambda[4])
{
  lambda[0] = positive(param_jet_lam0);
  lambda[1] = positive(param_jet_lam1);
  lambda[2] = positive(param_jet_lam2);
  lambda[3] = positive(param_jet_lam3);
}

static void add_tensor_component(double h[4][4], double weight,
                                 const double lambda[4],
                                 const double q0[4], const double q1[4],
                                 const double q2[4], const double q3[4])
{
  int i, j;

  if (weight <= 0.0 || !isfinite(weight)) return;

  for (i = 0; i < 4; i++) {
    for (j = 0; j < 4; j++) {
      h[i][j] += weight * (
          lambda[0]*lambda[0]*q0[i]*q0[j]
        + lambda[1]*lambda[1]*q1[i]*q1[j]
        + lambda[2]*lambda[2]*q2[i]*q2[j]
        + lambda[3]*lambda[3]*q3[i]*q3[j]);
    }
  }
}

static void add_disk_tensor(double h[4][4], double weight,
                            double x0, double x1, double x2, double x3)
{
  double q0[4], q1[4], q2[4], q3[4];
  double lambda[4];

  set_disk_advection_vector(q0, x0, x1, x2, x3);
  set_disk_triad(q1, q2, q3, x1, x2, x3);
  disk_corr_lengths(lambda, x0, x1, x2, x3);

  add_tensor_component(h, weight, lambda, q0, q1, q2, q3);
}

static void add_jet_tensor(double h[4][4], double weight,
                           double x0, double x1, double x2, double x3)
{
  double q0[4], q1[4], q2[4], q3[4];
  double lambda[4];

  set_jet_advection_vector(q0, x0, x1, x2, x3);
  set_jet_triad(q1, q2, q3, x1, x2, x3);
  jet_corr_lengths(lambda);

  add_tensor_component(h, weight, lambda, q0, q1, q2, q3);
}

static void set_h(double h[4][4], double x0, double x1, double x2, double x3)
{
  int i, j;

  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      h[i][j] = 0.0;

  if (param_component == 1) {
    add_jet_tensor(h, 1.0, x0, x1, x2, x3);
  } else if (param_component == 2) {
    double wd, wj;
    composite_weights(&wd, &wj, x1, x2, x3);
    add_disk_tensor(h, wd, x0, x1, x2, x3);
    add_jet_tensor(h, wj, x0, x1, x2, x3);
  } else {
    add_disk_tensor(h, 1.0, x0, x1, x2, x3);
  }
}

/* dh[i][j][k] = d h[i][j] / dx[k] */
static void set_dh(double dh[][4][4], double x0, double x1, double x2, double x3,
                   double dx0, double dx1, double dx2, double dx3)
{
  int i, j, k;
  double dx[4] = {dx0, dx1, dx2, dx3};
  double hm[4][4][4], hp[4][4][4];

  set_h(hm[0], x0 - dx0, x1, x2, x3);
  set_h(hp[0], x0 + dx0, x1, x2, x3);
  set_h(hm[1], x0, x1 - dx1, x2, x3);
  set_h(hp[1], x0, x1 + dx1, x2, x3);
  set_h(hm[2], x0, x1, x2 - dx2, x3);
  set_h(hp[2], x0, x1, x2 + dx2, x3);
  set_h(hm[3], x0, x1, x2, x3 - dx3);
  set_h(hp[3], x0, x1, x2, x3 + dx3);

  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      for (k = 0; k < 4; k++)
        dh[i][j][k] = 0.5 * (hp[k][i][j] - hm[k][i][j]) / dx[k];
}

static double ksq(double x0, double x1, double x2, double x3)
{
  (void) x0;
  (void) x1;
  (void) x2;
  (void) x3;
  return 1.0;
}

void param_coeff(double* coeff, double x0, double x1, double x2, double x3,
                 double dx0, double dx1, double dx2, double dx3, int index)
{
  (void) index;

  double h[4][4], dh[4][4][4];
  set_h(h, x0, x1, x2, x3);
  set_dh(dh, x0, x1, x2, x3, dx0, dx1, dx2, dx3);

  coeff[0] = ksq(x0, x1, x2, x3)
           + 2.0*(h[0][0]/dx0/dx0 + h[1][1]/dx1/dx1
                + h[2][2]/dx2/dx2 + h[3][3]/dx3/dx3);

  coeff[1] =  0.5*(-2.0*h[3][3] + dx3*(dh[3][3][3] + dh[2][3][2]
              + dh[1][3][1] + dh[0][3][0]))/(dx3*dx3);
  coeff[2] = -0.5*( 2.0*h[3][3] + dx3*(dh[3][3][3] + dh[2][3][2]
              + dh[1][3][1] + dh[0][3][0]))/(dx3*dx3);

  coeff[3] =  0.5*(-2.0*h[2][2] + dx2*(dh[3][2][3] + dh[2][2][2]
              + dh[1][2][1] + dh[0][2][0]))/(dx2*dx2);
  coeff[4] = -0.5*( 2.0*h[2][2] + dx2*(dh[3][2][3] + dh[2][2][2]
              + dh[1][2][1] + dh[0][2][0]))/(dx2*dx2);

  coeff[5] =  0.5*(-2.0*h[1][1] + dx1*(dh[3][1][3] + dh[2][1][2]
              + dh[1][1][1] + dh[0][1][0]))/(dx1*dx1);
  coeff[6] = -0.5*( 2.0*h[1][1] + dx1*(dh[3][1][3] + dh[2][1][2]
              + dh[1][1][1] + dh[0][1][0]))/(dx1*dx1);

  coeff[7] =  0.5*(-2.0*h[0][0] + dx0*(dh[3][0][3] + dh[2][0][2]
              + dh[1][0][1] + dh[0][0][0]))/(dx0*dx0);
  coeff[8] = -0.5*( 2.0*h[0][0] + dx0*(dh[3][0][3] + dh[2][0][2]
              + dh[1][0][1] + dh[0][0][0]))/(dx0*dx0);

  coeff[9]  = 0.5*h[2][3] / (dx2 * dx3);
  coeff[10] = 0.5*h[1][3] / (dx1 * dx3);
  coeff[11] = 0.5*h[1][2] / (dx1 * dx2);
  coeff[12] = 0.5*h[0][3] / (dx0 * dx3);
  coeff[13] = 0.5*h[0][2] / (dx0 * dx2);
  coeff[14] = 0.5*h[0][1] / (dx0 * dx1);
}


static double det4(double m[4][4])
{
  int i, j, k, pivot;
  double a[4][4];
  double det = 1.0;
  double sign = 1.0;

  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      a[i][j] = m[i][j];

  for (k = 0; k < 4; k++) {
    double maxabs = fabs(a[k][k]);
    pivot = k;
    for (i = k + 1; i < 4; i++) {
      const double val = fabs(a[i][k]);
      if (val > maxabs) {
        maxabs = val;
        pivot = i;
      }
    }

    if (maxabs < 1.e-300 || !isfinite(maxabs))
      return 0.0;

    if (pivot != k) {
      for (j = k; j < 4; j++) {
        const double tmp = a[k][j];
        a[k][j] = a[pivot][j];
        a[pivot][j] = tmp;
      }
      sign = -sign;
    }

    det *= a[k][k];
    for (i = k + 1; i < 4; i++) {
      const double factor = a[i][k] / a[k][k];
      for (j = k + 1; j < 4; j++)
        a[i][j] -= factor * a[k][j];
    }
  }

  det *= sign;
  return isfinite(det) ? det : 0.0;
}

void param_set_source(double* values, gsl_rng* rstate, int ni, int nj, int nk, int nl,
                      int pi, int pj, int pk, int pl, int npi, int npj, int npk, int npl,
                      double dx0, double dx1, double dx2, double dx3, int nrecur)
{
  (void) npi;
  (void) npj;
  (void) npk;
  (void) npl;
  (void) nrecur;

  const int nvalues = ni * nj * nk * nl;
  const double Nnu4_sq = 96.0 * M_PI * M_PI; /* (4 pi sqrt(6))^2 for nu = 2, d = 4 */
  int i;

  for (i = 0; i < nvalues; i++) {
    int gridk = i / (ni * nj * nl);
    int gridl = (i - gridk * ni * nj * nl) / (nj * ni);
    int gridj = (i - gridk * ni * nj * nl - gridl * nj * ni) / ni;
    int gridi = i - gridk * ni * nj * nl - gridl * nj * ni - gridj * ni;

    gridk += pk * nk;
    gridl += pl * nl;
    gridj += pj * nj;
    gridi += pi * ni;

    const double x0 = param_x0start + dx0 * (gridk + 0.5);
    const double x1 = param_x1start + dx1 * (gridl + 0.5);
    const double x2 = param_x2start + dx2 * (gridj + 0.5);
    const double x3 = param_x3start + dx3 * (gridi + 0.5);

    double h[4][4];
    set_h(h, x0, x1, x2, x3);

    const double cell_vol = dx0 * dx1 * dx2 * dx3;
    const double det_h = fmax(det4(h), 0.0);
    const double scaling_sq = Nnu4_sq * sqrt(det_h) / cell_vol;
    const double scaling = fmax(sqrt(fmax(scaling_sq, 0.0)), SMALL);

    values[i] = gsl_ran_gaussian_ziggurat(rstate, 1.0) * scaling;
  }
}
/* Composite-tensor controls.  If normalize_weights=0, the raw GRF is naturally
   concentrated in the torus and jet.  If normalize_weights=1, the weights only
   interpolate the orientation/correlation tensor and do not localize the field. */
static double blend_floor = 1.e-4;
static int normalize_weights = 0;

/* Kerr four-velocity parameters. */
static double spin = 0.94;
static double sub_kep = 1.0;       /* xi */
static double ell_delta = 0.0;     /* delta */
static int orbit_sigma = 1;        /* +1 prograde, -1 retrograde */
static double betar = 1.0;
static int use_plunge = 1;
static int enforce_timelike = 1;

/* Simple jet advection model: v = jet_vpol e_r + rho jet_omega e_phi. */
static double jet_vpol = 0.5;
static double jet_omega = 0.0;

static double clamp(double x, double lo, double hi)
{
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

static double positive(double x)
{
  if (!isfinite(x)) return LAMBDA_FLOOR;
  return fmax(fabs(x), LAMBDA_FLOOR);
}

static void read_double_if_exists(const char *name, double *val)
{
  if (hdf5_exists(name)) hdf5_read_single_val(val, name, H5T_IEEE_F64LE);
}

static void read_int_if_exists(const char *name, int *val)
{
  if (hdf5_exists(name)) hdf5_read_single_val(val, name, H5T_STD_I32LE);
}

/* Read in parameters from input file.  Missing datasets are ignored so old
   parameter files remain usable. */
void param_read_params(char* filename)
{
  if (filename != NULL) {
    hdf5_open(filename);
    hdf5_set_directory("/params/");

    read_int_if_exists("component", &param_component);

    read_double_if_exists("lam0", &param_lam0);
    read_double_if_exists("lam1", &param_lam1);
    read_double_if_exists("lam2", &param_lam2);
    read_double_if_exists("lam3", &param_lam3);

    read_double_if_exists("jet_lam0", &param_jet_lam0);
    read_double_if_exists("jet_lam1", &param_jet_lam1);
    read_double_if_exists("jet_lam2", &param_jet_lam2);
    read_double_if_exists("jet_lam3", &param_jet_lam3);

    read_double_if_exists("pitch", &param_pitch);
    read_double_if_exists("jet_chi0", &param_jet_chi0);
    read_double_if_exists("jet_chi_rho", &param_jet_chi_rho);

    read_double_if_exists("torus_r0", &torus_r0);
    read_double_if_exists("torus_ar", &torus_ar);
    read_double_if_exists("torus_az", &torus_az);
    read_double_if_exists("jet_width", &jet_width);
    read_double_if_exists("jet_opening", &jet_opening);
    read_double_if_exists("blend_floor", &blend_floor);
    read_int_if_exists("normalize_weights", &normalize_weights);

    read_double_if_exists("spin", &spin);
    read_double_if_exists("xi", &sub_kep);
    read_double_if_exists("delta", &ell_delta);
    read_int_if_exists("sigma", &orbit_sigma);
    read_double_if_exists("betar", &betar);
    read_int_if_exists("use_plunge", &use_plunge);
    read_int_if_exists("enforce_timelike", &enforce_timelike);

    read_double_if_exists("jet_vpol", &jet_vpol);
    read_double_if_exists("jet_omega", &jet_omega);

    hdf5_close();
  }

  if (orbit_sigma >= 0) orbit_sigma = 1;
  else orbit_sigma = -1;
  if (param_component < 0 || param_component > 2) param_component = 2;
  spin = clamp(spin, -0.999999, 0.999999);
  betar = clamp(betar, 0.0, 1.0);
  torus_r0 = positive(torus_r0);
  torus_ar = positive(torus_ar);
  torus_az = positive(torus_az);
  jet_width = positive(jet_width);
  jet_opening = fmax(0.0, jet_opening);
  blend_floor = fmax(0.0, blend_floor);
  normalize_weights = normalize_weights ? 1 : 0;
}

/* Write out parameters to output file. */
void param_write_params(char* filename)
{
  (void) filename;

  hdf5_set_directory("/params/");

  hdf5_write_single_val(&param_component, "component", H5T_STD_I32LE);

  hdf5_write_single_val(&param_lam0, "lam0", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_lam1, "lam1", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_lam2, "lam2", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_lam3, "lam3", H5T_IEEE_F64LE);

  hdf5_write_single_val(&param_jet_lam0, "jet_lam0", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_jet_lam1, "jet_lam1", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_jet_lam2, "jet_lam2", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_jet_lam3, "jet_lam3", H5T_IEEE_F64LE);

  hdf5_write_single_val(&param_pitch, "pitch", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_jet_chi0, "jet_chi0", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_jet_chi_rho, "jet_chi_rho", H5T_IEEE_F64LE);

  hdf5_write_single_val(&torus_r0, "torus_r0", H5T_IEEE_F64LE);
  hdf5_write_single_val(&torus_ar, "torus_ar", H5T_IEEE_F64LE);
  hdf5_write_single_val(&torus_az, "torus_az", H5T_IEEE_F64LE);
  hdf5_write_single_val(&jet_width, "jet_width", H5T_IEEE_F64LE);
  hdf5_write_single_val(&jet_opening, "jet_opening", H5T_IEEE_F64LE);
  hdf5_write_single_val(&blend_floor, "blend_floor", H5T_IEEE_F64LE);
  hdf5_write_single_val(&normalize_weights, "normalize_weights", H5T_STD_I32LE);

  hdf5_write_single_val(&spin, "spin", H5T_IEEE_F64LE);
  hdf5_write_single_val(&sub_kep, "xi", H5T_IEEE_F64LE);
  hdf5_write_single_val(&ell_delta, "delta", H5T_IEEE_F64LE);
  hdf5_write_single_val(&orbit_sigma, "sigma", H5T_STD_I32LE);
  hdf5_write_single_val(&betar, "betar", H5T_IEEE_F64LE);
  hdf5_write_single_val(&use_plunge, "use_plunge", H5T_STD_I32LE);
  hdf5_write_single_val(&enforce_timelike, "enforce_timelike", H5T_STD_I32LE);

  hdf5_write_single_val(&jet_vpol, "jet_vpol", H5T_IEEE_F64LE);
  hdf5_write_single_val(&jet_omega, "jet_omega", H5T_IEEE_F64LE);
}

void param_set_output_name(char* filename, size_t filename_size,
                           int ni, int nj, int nk, int nl,
                           int npi, int npj, int npk, int npl, char* dir)
{
  time_t rawtime;
  struct tm *timeinfo;
  char buffer[255];
  const char *component = (param_component == 1) ? "jet" : ((param_component == 2) ? "torusjet" : "disk");

  (void) ni;
  (void) nl;
  (void) npi;
  (void) npl;

  time(&rawtime);
  timeinfo = localtime(&rawtime);
  strftime(buffer, 255, "%Y_%m_%d_%H%M%S", timeinfo);

  int n = snprintf(filename, filename_size,
                   "%s/%s_%s_%s_%d_%d_%.0f_%.0f_a%.4f_xi%.2f_delta%.2f_br%.2f_l%.2f_%.2f_%.2f.h5",
                   dir, model_name, component, buffer, npj * nj, npk * nk,
                   param_x1end, param_x0end, spin, sub_kep, ell_delta, betar,
                   param_lam1, param_lam2, param_lam3);
  if (n < 0 || (size_t)n >= filename_size) {
    fprintf(stderr, "Output filename is too long; use a shorter output directory or filename template.\n");
    exit(EXIT_FAILURE);
  }
}

typedef struct {
  double r;
  double rho;
  double sinth;
  double costh;
  double phi;
  double xnorm;
  double ynorm;
  double znorm;
} bl_point;

typedef struct {
  double delta;
  double sigma;
  double A;
  double gtt;
  double gtphi;
  double grr;
  double gphiphi;
  double gtt_inv;
  double gtphi_inv;
  double grr_inv;
  double gphiphi_inv;
} kerr_metric;

static double horizon_radius(double a)
{
  return 1.0 + sqrt(fmax(0.0, 1.0 - a*a));
}

static void set_bl_point(bl_point *p, double x, double y, double z)
{
  const double r_raw = sqrt(x*x + y*y + z*z);
  const double rho = sqrt(x*x + y*y);
  const double r_min = horizon_radius(spin) + 1.e-6;

  p->rho = rho;
  p->phi = atan2(y, x);

  if (r_raw > SMALL) {
    p->sinth = rho / r_raw;
    p->costh = z / r_raw;
    p->xnorm = x / r_raw;
    p->ynorm = y / r_raw;
    p->znorm = z / r_raw;
  } else {
    p->sinth = 1.0;
    p->costh = 0.0;
    p->xnorm = 0.0;
    p->ynorm = 0.0;
    p->znorm = 0.0;
  }

  p->r = fmax(r_raw, r_min);
}

static void set_kerr_metric(kerr_metric *m, double r, double sinth, double costh, double a)
{
  double sin2 = fmax(sinth*sinth, 1.e-14);
  double cos2 = costh*costh;
  double Sigma = r*r + a*a*cos2;

  m->delta = r*r - 2.0*r + a*a;
  m->delta = fmax(m->delta, 1.e-14);
  m->A = (r*r + a*a)*(r*r + a*a) - a*a*m->delta*sin2;
  m->sigma = Sigma;

  m->gtt = -(1.0 - 2.0*r/Sigma);
  m->gtphi = -2.0*a*r*sin2/Sigma;
  m->grr = Sigma/m->delta;
  m->gphiphi = m->A*sin2/Sigma;

  m->gtt_inv = -m->A/(Sigma*m->delta);
  m->gtphi_inv = -2.0*a*r/(Sigma*m->delta);
  m->grr_inv = m->delta/Sigma;
  m->gphiphi_inv = (m->delta - a*a*sin2)/(Sigma*m->delta*sin2);
}

static double r_isco(double a, int sigma)
{
  const double aa = fabs(a);
  const double z1 = 1.0 + pow(1.0 - aa*aa, 1.0/3.0)
                    * (pow(1.0 + aa, 1.0/3.0) + pow(1.0 - aa, 1.0/3.0));
  const double z2 = sqrt(3.0*aa*aa + z1*z1);
  const double branch = (sigma >= 0) ? -1.0 : 1.0;
  return 3.0 + z2 + branch * sqrt((3.0 - z1)*(3.0 + z1 + 2.0*z2));
}

static double ell_profile(double rho)
{
  const double r = fmax(rho, 1.e-6);
  const double sr = sqrt(r);
  const double sigma = (orbit_sigma >= 0) ? 1.0 : -1.0;
  const double a = spin;
  double denom = r - 2.0 + ell_delta + sigma*a/sr;

  if (fabs(denom) < 1.e-10)
    denom = (denom >= 0.0) ? 1.e-10 : -1.e-10;

  return sub_kep * sigma * (r*sr - 2.0*sigma*a + a*a/sr) / denom;
}

static double U_of_ell(double ell, const kerr_metric *m)
{
  return -m->gtt_inv + 2.0*ell*m->gtphi_inv - ell*ell*m->gphiphi_inv;
}

static double omega_from_ell(double ell, const kerr_metric *m)
{
  const double denom = m->gtt_inv - ell*m->gtphi_inv;
  if (fabs(denom) < SMALL) return 0.0;
  return (m->gtphi_inv - ell*m->gphiphi_inv) / denom;
}

static double clamp_omega(double omega, const kerr_metric *m)
{
  if (!enforce_timelike || m->gphiphi <= SMALL) return omega;

  double disc = m->gtphi*m->gtphi - m->gtt*m->gphiphi;
  disc = sqrt(fmax(disc, 0.0));

  double om1 = (-m->gtphi - disc) / m->gphiphi;
  double om2 = (-m->gtphi + disc) / m->gphiphi;
  double omin = fmin(om1, om2);
  double omax = fmax(om1, om2);
  double eps = 1.e-8 * fmax(1.0, omax - omin);

  return clamp(omega, omin + eps, omax - eps);
}

static double D_of_omega(double omega, const kerr_metric *m)
{
  return -m->gtt - 2.0*m->gtphi*omega - m->gphiphi*omega*omega;
}

static double freefall_ur(const kerr_metric *m)
{
  return -sqrt(fmax(0.0, (-1.0 - m->gtt_inv) * m->grr_inv));
}

static double plunge_hat_ur(const bl_point *p, const kerr_metric *m)
{
  if (!use_plunge) return 0.0;

  const double rms = r_isco(spin, orbit_sigma);
  if (p->r >= rms) return 0.0;

  const double rb = rms;
  const double sinb = p->sinth;
  const double cosb = p->costh;
  const double rho_b = rb * sinb;
  const double ell_b = ell_profile(rho_b);

  kerr_metric mb;
  set_kerr_metric(&mb, rb, sinb, cosb, spin);

  const double Ub = U_of_ell(ell_b, &mb);
  if (Ub <= SMALL || !isfinite(Ub)) return 0.0;

  const double Eb = 1.0 / sqrt(Ub);
  const double Lb = Eb * ell_b;
  const double Rpl = -1.0 - m->gtt_inv*Eb*Eb
                     + 2.0*m->gtphi_inv*Eb*Lb
                     - m->gphiphi_inv*Lb*Lb;

  if (Rpl <= 0.0 || !isfinite(Rpl)) return 0.0;
  return -sqrt(fmax(0.0, m->grr_inv * Rpl));
}

static void disk_velocity(double v[3], double x0, double x1, double x2, double x3)
{
  (void) x0;

  bl_point p;
  kerr_metric m;
  set_bl_point(&p, x1, x2, x3);
  set_kerr_metric(&m, p.r, p.sinth, p.costh, spin);

  double ell = ell_profile(fmax(p.rho, 1.e-6));
  double omega = clamp_omega(omega_from_ell(ell, &m), &m);

  double hat_ur = plunge_hat_ur(&p, &m);
  double bar_ur = freefall_ur(&m);
  double ur = betar * hat_ur + (1.0 - betar) * bar_ur;
  double D = D_of_omega(omega, &m);

  if (D <= SMALL || !isfinite(D)) {
    omega = 0.0;
    ur = 0.0;
    D = D_of_omega(omega, &m);
  }

  const double ut = sqrt(fmax(SMALL, (1.0 + m.grr*ur*ur) / fmax(D, SMALL)));
  const double vr = ur / ut;

  v[0] = vr*p.xnorm - omega*x2;
  v[1] = vr*p.ynorm + omega*x1;
  v[2] = vr*p.znorm;
}

static double disk_omega(double x0, double x1, double x2, double x3)
{
  (void) x0;

  bl_point p;
  kerr_metric m;
  set_bl_point(&p, x1, x2, x3);
  set_kerr_metric(&m, p.r, p.sinth, p.costh, spin);

  return clamp_omega(omega_from_ell(ell_profile(fmax(p.rho, 1.e-6)), &m), &m);
}

static void jet_velocity(double v[3], double x0, double x1, double x2, double x3)
{
  (void) x0;

  const double r = sqrt(x1*x1 + x2*x2 + x3*x3);
  double er[3] = {0.0, 0.0, (x3 >= 0.0) ? 1.0 : -1.0};

  if (r > SMALL) {
    er[0] = x1 / r;
    er[1] = x2 / r;
    er[2] = x3 / r;
  }

  v[0] = jet_vpol*er[0] - jet_omega*x2;
  v[1] = jet_vpol*er[1] + jet_omega*x1;
  v[2] = jet_vpol*er[2];
}

static void set_q0_from_velocity(double q0[4], const double v[3])
{
  q0[0] = 1.0;
  q0[1] = v[0];
  q0[2] = v[1];
  q0[3] = v[2];
}

static void set_disk_advection_vector(double q0[4], double x0, double x1, double x2, double x3)
{
  double v[3];
  disk_velocity(v, x0, x1, x2, x3);
  set_q0_from_velocity(q0, v);
}

static void set_jet_advection_vector(double q0[4], double x0, double x1, double x2, double x3)
{
  double v[3];
  jet_velocity(v, x0, x1, x2, x3);
  set_q0_from_velocity(q0, v);
}

static void normalize3(double v[3], const double fallback[3])
{
  const double n = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  if (n > SMALL && isfinite(n)) {
    v[0] /= n;
    v[1] /= n;
    v[2] /= n;
  } else {
    v[0] = fallback[0];
    v[1] = fallback[1];
    v[2] = fallback[2];
  }
}

static void set_cylindrical_frame(double erho[3], double ephi[3], double x1, double x2)
{
  const double rho = sqrt(x1*x1 + x2*x2);

  erho[0] = 1.0;
  erho[1] = 0.0;
  erho[2] = 0.0;

  ephi[0] = 0.0;
  ephi[1] = 1.0;
  ephi[2] = 0.0;

  if (rho > SMALL) {
    erho[0] = x1 / rho;
    erho[1] = x2 / rho;
    erho[2] = 0.0;

    ephi[0] = -x2 / rho;
    ephi[1] = x1 / rho;
    ephi[2] = 0.0;
  }
}

static double torus_weight(double x1, double x2, double x3)
{
  const double rho = sqrt(x1*x1 + x2*x2);
  const double u = (rho - torus_r0) / fmax(torus_ar, LAMBDA_FLOOR);
  const double v = x3 / fmax(torus_az, LAMBDA_FLOOR);

  return exp(-0.5 * (u*u + v*v));
}

static double collimated_jet_weight(double x1, double x2, double x3)
{
  const double rho = sqrt(x1*x1 + x2*x2);
  const double width = fmax(jet_width + jet_opening*fabs(x3), LAMBDA_FLOOR);
  return exp(-0.5 * rho*rho / (width*width));
}

static void composite_weights(double *wd, double *wj, double x1, double x2, double x3)
{
  double d = torus_weight(x1, x2, x3);
  double j = collimated_jet_weight(x1, x2, x3);

  if (normalize_weights) {
    const double sum = d + j + 2.0*blend_floor;
    *wd = (d + blend_floor) / fmax(sum, SMALL);
    *wj = (j + blend_floor) / fmax(sum, SMALL);
  } else {
    *wd = d + blend_floor;
    *wj = j + blend_floor;
  }
}

static void set_disk_triad(double e1[4], double e2[4], double e3[4],
                           double x1, double x2, double x3)
{
  const double rho = sqrt(x1*x1 + x2*x2);
  const double u = (rho - torus_r0) / fmax(torus_ar, LAMBDA_FLOOR);
  const double v = x3 / fmax(torus_az, LAMBDA_FLOOR);

  double erho[3], ephi[3];
  double epol[3], enorm[3];
  const double fallback_pol[3] = {0.0, 0.0, 1.0};
  double fallback_norm[3] = {1.0, 0.0, 0.0};

  set_cylindrical_frame(erho, ephi, x1, x2);
  fallback_norm[0] = erho[0];
  fallback_norm[1] = erho[1];
  fallback_norm[2] = erho[2];

  /* Elliptical torus cross-section.  epol is tangent to the meridional
     torus cross-section; enorm is the corresponding cross-section normal. */
  epol[0] = -torus_ar * v * erho[0];
  epol[1] = -torus_ar * v * erho[1];
  epol[2] =  torus_az * u;

  enorm[0] =  torus_az * u * erho[0];
  enorm[1] =  torus_az * u * erho[1];
  enorm[2] =  torus_ar * v;

  normalize3(epol, fallback_pol);
  normalize3(enorm, fallback_norm);

  const double cp = cos(param_pitch);
  const double sp = sin(param_pitch);

  e1[0] = e2[0] = e3[0] = 0.0;

  /* Long disk correlations follow a spiral/helical direction on the torus
     surface rather than a purely cylindrical spiral at fixed z. */
  e1[1] = cp*ephi[0] + sp*epol[0];
  e1[2] = cp*ephi[1] + sp*epol[1];
  e1[3] = cp*ephi[2] + sp*epol[2];

  e2[1] = -sp*ephi[0] + cp*epol[0];
  e2[2] = -sp*ephi[1] + cp*epol[1];
  e2[3] = -sp*ephi[2] + cp*epol[2];

  e3[1] = enorm[0];
  e3[2] = enorm[1];
  e3[3] = enorm[2];
}

static void set_jet_triad(double e1[4], double e2[4], double e3[4],
                          double x1, double x2, double x3)
{
  const double rho = sqrt(x1*x1 + x2*x2);
  const double r = sqrt(rho*rho + x3*x3);

  double er[3] = {0.0, 0.0, (x3 >= 0.0) ? 1.0 : -1.0};
  double eth[3] = {1.0, 0.0, 0.0};
  double eph[3] = {0.0, 1.0, 0.0};

  if (rho > SMALL) {
    eph[0] = -x2 / rho;
    eph[1] = x1 / rho;
    eph[2] = 0.0;
  }

  if (r > SMALL && rho > SMALL) {
    er[0] = x1 / r;
    er[1] = x2 / r;
    er[2] = x3 / r;

    eth[0] = x1*x3/(rho*r);
    eth[1] = x2*x3/(rho*r);
    eth[2] = -rho/r;
  }

  const double rho_chi = fmax(param_jet_chi_rho, SMALL);
  const double chi = param_jet_chi0 * (1.0 - exp(-rho*rho/(rho_chi*rho_chi)));
  const double cc = cos(chi);
  const double sc = sin(chi);

  e1[0] = e2[0] = e3[0] = 0.0;

  e1[1] = cc*er[0] + sc*eph[0];
  e1[2] = cc*er[1] + sc*eph[1];
  e1[3] = cc*er[2] + sc*eph[2];

  e2[1] = eth[0];
  e2[2] = eth[1];
  e2[3] = eth[2];

  e3[1] = -sc*er[0] + cc*eph[0];
  e3[2] = -sc*er[1] + cc*eph[1];
  e3[3] = -sc*er[2] + cc*eph[2];
}

static double disk_corr_time(double x0, double x1, double x2, double x3)
{
  if (param_lam0 > 0.0)
    return positive(param_lam0);

  const double omega = fabs(disk_omega(x0, x1, x2, x3));
  return positive(2.0 * M_PI / fmax(omega, OMEGA_FLOOR));
}

static void disk_corr_lengths(double lambda[4], double x0, double x1, double x2, double x3)
{
  (void) x1;
  (void) x2;

  lambda[0] = disk_corr_time(x0, x1, x2, x3);
  lambda[1] = positive(param_lam1);
  lambda[2] = positive(param_lam2);
  lambda[3] = positive(param_lam3);
}

static void jet_corr_lengths(double lambda[4])
{
  lambda[0] = positive(param_jet_lam0);
  lambda[1] = positive(param_jet_lam1);
  lambda[2] = positive(param_jet_lam2);
  lambda[3] = positive(param_jet_lam3);
}

static void add_tensor_component(double h[4][4], double weight,
                                 const double lambda[4],
                                 const double q0[4], const double q1[4],
                                 const double q2[4], const double q3[4])
{
  int i, j;

  if (weight <= 0.0 || !isfinite(weight)) return;

  for (i = 0; i < 4; i++) {
    for (j = 0; j < 4; j++) {
      h[i][j] += weight * (
          lambda[0]*lambda[0]*q0[i]*q0[j]
        + lambda[1]*lambda[1]*q1[i]*q1[j]
        + lambda[2]*lambda[2]*q2[i]*q2[j]
        + lambda[3]*lambda[3]*q3[i]*q3[j]);
    }
  }
}

static void add_disk_tensor(double h[4][4], double weight,
                            double x0, double x1, double x2, double x3)
{
  double q0[4], q1[4], q2[4], q3[4];
  double lambda[4];

  set_disk_advection_vector(q0, x0, x1, x2, x3);
  set_disk_triad(q1, q2, q3, x1, x2, x3);
  disk_corr_lengths(lambda, x0, x1, x2, x3);

  add_tensor_component(h, weight, lambda, q0, q1, q2, q3);
}

static void add_jet_tensor(double h[4][4], double weight,
                           double x0, double x1, double x2, double x3)
{
  double q0[4], q1[4], q2[4], q3[4];
  double lambda[4];

  set_jet_advection_vector(q0, x0, x1, x2, x3);
  set_jet_triad(q1, q2, q3, x1, x2, x3);
  jet_corr_lengths(lambda);

  add_tensor_component(h, weight, lambda, q0, q1, q2, q3);
}

static void set_h(double h[4][4], double x0, double x1, double x2, double x3)
{
  int i, j;

  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      h[i][j] = 0.0;

  if (param_component == 1) {
    add_jet_tensor(h, 1.0, x0, x1, x2, x3);
  } else if (param_component == 2) {
    double wd, wj;
    composite_weights(&wd, &wj, x1, x2, x3);
    add_disk_tensor(h, wd, x0, x1, x2, x3);
    add_jet_tensor(h, wj, x0, x1, x2, x3);
  } else {
    add_disk_tensor(h, 1.0, x0, x1, x2, x3);
  }
}

/* dh[i][j][k] = d h[i][j] / dx[k] */
static void set_dh(double dh[][4][4], double x0, double x1, double x2, double x3,
                   double dx0, double dx1, double dx2, double dx3)
{
  int i, j, k;
  double dx[4] = {dx0, dx1, dx2, dx3};
  double hm[4][4][4], hp[4][4][4];

  set_h(hm[0], x0 - dx0, x1, x2, x3);
  set_h(hp[0], x0 + dx0, x1, x2, x3);
  set_h(hm[1], x0, x1 - dx1, x2, x3);
  set_h(hp[1], x0, x1 + dx1, x2, x3);
  set_h(hm[2], x0, x1, x2 - dx2, x3);
  set_h(hp[2], x0, x1, x2 + dx2, x3);
  set_h(hm[3], x0, x1, x2, x3 - dx3);
  set_h(hp[3], x0, x1, x2, x3 + dx3);

  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      for (k = 0; k < 4; k++)
        dh[i][j][k] = 0.5 * (hp[k][i][j] - hm[k][i][j]) / dx[k];
}

static double ksq(double x0, double x1, double x2, double x3)
{
  (void) x0;
  (void) x1;
  (void) x2;
  (void) x3;
  return 1.0;
}

void param_coeff(double* coeff, double x0, double x1, double x2, double x3,
                 double dx0, double dx1, double dx2, double dx3, int index)
{
  (void) index;

  double h[4][4], dh[4][4][4];
  set_h(h, x0, x1, x2, x3);
  set_dh(dh, x0, x1, x2, x3, dx0, dx1, dx2, dx3);

  coeff[0] = ksq(x0, x1, x2, x3)
           + 2.0*(h[0][0]/dx0/dx0 + h[1][1]/dx1/dx1
                + h[2][2]/dx2/dx2 + h[3][3]/dx3/dx3);

  coeff[1] =  0.5*(-2.0*h[3][3] + dx3*(dh[3][3][3] + dh[2][3][2]
              + dh[1][3][1] + dh[0][3][0]))/(dx3*dx3);
  coeff[2] = -0.5*( 2.0*h[3][3] + dx3*(dh[3][3][3] + dh[2][3][2]
              + dh[1][3][1] + dh[0][3][0]))/(dx3*dx3);

  coeff[3] =  0.5*(-2.0*h[2][2] + dx2*(dh[3][2][3] + dh[2][2][2]
              + dh[1][2][1] + dh[0][2][0]))/(dx2*dx2);
  coeff[4] = -0.5*( 2.0*h[2][2] + dx2*(dh[3][2][3] + dh[2][2][2]
              + dh[1][2][1] + dh[0][2][0]))/(dx2*dx2);

  coeff[5] =  0.5*(-2.0*h[1][1] + dx1*(dh[3][1][3] + dh[2][1][2]
              + dh[1][1][1] + dh[0][1][0]))/(dx1*dx1);
  coeff[6] = -0.5*( 2.0*h[1][1] + dx1*(dh[3][1][3] + dh[2][1][2]
              + dh[1][1][1] + dh[0][1][0]))/(dx1*dx1);

  coeff[7] =  0.5*(-2.0*h[0][0] + dx0*(dh[3][0][3] + dh[2][0][2]
              + dh[1][0][1] + dh[0][0][0]))/(dx0*dx0);
  coeff[8] = -0.5*( 2.0*h[0][0] + dx0*(dh[3][0][3] + dh[2][0][2]
              + dh[1][0][1] + dh[0][0][0]))/(dx0*dx0);

  coeff[9]  = 0.5*h[2][3] / (dx2 * dx3);
  coeff[10] = 0.5*h[1][3] / (dx1 * dx3);
  coeff[11] = 0.5*h[1][2] / (dx1 * dx2);
  coeff[12] = 0.5*h[0][3] / (dx0 * dx3);
  coeff[13] = 0.5*h[0][2] / (dx0 * dx2);
  coeff[14] = 0.5*h[0][1] / (dx0 * dx1);
}


static double det4(double m[4][4])
{
  int i, j, k, pivot;
  double a[4][4];
  double det = 1.0;
  double sign = 1.0;

  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      a[i][j] = m[i][j];

  for (k = 0; k < 4; k++) {
    double maxabs = fabs(a[k][k]);
    pivot = k;
    for (i = k + 1; i < 4; i++) {
      const double val = fabs(a[i][k]);
      if (val > maxabs) {
        maxabs = val;
        pivot = i;
      }
    }

    if (maxabs < 1.e-300 || !isfinite(maxabs))
      return 0.0;

    if (pivot != k) {
      for (j = k; j < 4; j++) {
        const double tmp = a[k][j];
        a[k][j] = a[pivot][j];
        a[pivot][j] = tmp;
      }
      sign = -sign;
    }

    det *= a[k][k];
    for (i = k + 1; i < 4; i++) {
      const double factor = a[i][k] / a[k][k];
      for (j = k + 1; j < 4; j++)
        a[i][j] -= factor * a[k][j];
    }
  }

  det *= sign;
  return isfinite(det) ? det : 0.0;
}

void param_set_source(double* values, gsl_rng* rstate, int ni, int nj, int nk, int nl,
                      int pi, int pj, int pk, int pl, int npi, int npj, int npk, int npl,
                      double dx0, double dx1, double dx2, double dx3, int nrecur)
{
  (void) npi;
  (void) npj;
  (void) npk;
  (void) npl;
  (void) nrecur;

  const int nvalues = ni * nj * nk * nl;
  const double Nnu4_sq = 96.0 * M_PI * M_PI; /* (4 pi sqrt(6))^2 for nu = 2, d = 4 */
  int i;

  for (i = 0; i < nvalues; i++) {
    int gridk = i / (ni * nj * nl);
    int gridl = (i - gridk * ni * nj * nl) / (nj * ni);
    int gridj = (i - gridk * ni * nj * nl - gridl * nj * ni) / ni;
    int gridi = i - gridk * ni * nj * nl - gridl * nj * ni - gridj * ni;

    gridk += pk * nk;
    gridl += pl * nl;
    gridj += pj * nj;
    gridi += pi * ni;

    const double x0 = param_x0start + dx0 * (gridk + 0.5);
    const double x1 = param_x1start + dx1 * (gridl + 0.5);
    const double x2 = param_x2start + dx2 * (gridj + 0.5);
    const double x3 = param_x3start + dx3 * (gridi + 0.5);

    double h[4][4];
    set_h(h, x0, x1, x2, x3);

    const double cell_vol = dx0 * dx1 * dx2 * dx3;
    const double det_h = fmax(det4(h), 0.0);
    const double scaling_sq = Nnu4_sq * sqrt(det_h) / cell_vol;
    const double scaling = fmax(sqrt(fmax(scaling_sq, 0.0)), SMALL);

    values[i] = gsl_ran_gaussian_ziggurat(rstate, 1.0) * scaling;
  }
}
