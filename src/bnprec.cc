#include "bnprec.hh"

BNPRec::BNPRec(Env &env, Ratings &ratings)
  : _env(env), _ratings(ratings),
    _n(env.n), 
    _m(env.m),
    _k(env.k),
    _c(env.scale), _alpha(env.alpha),
    _iter(0),
    _start_time(time(0)),
    _beta_shape_prior(0.3), _beta_rate_prior(0.3),
    _beta("beta", _beta_shape_prior, _beta_rate_prior, _m, _k,&_r),
    _s("s", _alpha, _c, _n, &_r),
    _v(_n, _k),
    _zusers(_n,_k), _zitems(_m,_k), _user_budget(_n),
    _ebetasum(_k), _ethetasum(_k),
    _A(_n,_k),
    _scalar_finite_sums(_n),
    _pi(_n,_k), _theta(_n,_k),
    _logpi(_n,_k), _active_k(_k),
    _tau0(1024), _kappa(env.kappa),
    _save_ranking_file(false),
    _prev_h(0)
{
  gsl_rng_env_setup();
  const gsl_rng_type *T = gsl_rng_default;
  _r = gsl_rng_alloc(T);
  if (_env.seed)
    gsl_rng_set(_r, _env.seed);

  _af = fopen(Env::file_str("/logl.tsv").c_str(), "w");
  if (!_af)  {
    printf("cannot open logl file:%s\n",  strerror(errno));
    exit(-1);
  }

  _vf = fopen(Env::file_str("/validation.tsv").c_str(), "w");
  if (!_vf)  {
    printf("cannot open heldout file:%s\n",  strerror(errno));
    exit(-1);
  }

  _tf = fopen(Env::file_str("/test.tsv").c_str(), "w");
  if (!_vf)  {
    printf("cannot open heldout file:%s\n",  strerror(errno));
    exit(-1);
  }
  _pf = fopen(Env::file_str("/precision.txt").c_str(), "w");
  if (!_pf)  {
    printf("cannot open logl file:%s\n",  strerror(errno));
    exit(-1);
  }
  Env::plog("c", _c);
  Env::plog("beta_shape_prior", _beta_shape_prior);
  Env::plog("beta_rate_prior", _beta_rate_prior);
}

void
BNPRec::initialize()
{
  load_validation_and_test_sets();
  _beta.initialize();
  _s.initialize();
  initialize_sticks();
}

BNPRec::~BNPRec()
{
  fclose(_vf);
  fclose(_af);
  fclose(_pf);
  fclose(_tf);
}

void
BNPRec::initialize_sticks()
{
  double **vd = _v.data();
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t k = 0; k < _k; ++k)  {
      //vd[i][k] = gsl_ran_beta(_r, 1, _alpha);
      vd[i][k] = 0.001 * gsl_rng_uniform(_r);
    }
  compute_pi();
}

//
// Batch inference algorithm (VB or using natural gradients)
//
void
BNPRec::batch_infer()
{
  initialize();
  compute_all_expectations();
  recompute_sums();
  set_rho();

  Array phi(_k);

  double lsum = 0;
  while(1) {
    clear_state();

    for (uint32_t u = 0; u < _n; ++u) {
      const vector<uint32_t> *movies = _ratings.get_movies(u); 

      for (uint32_t j = 0; movies && j < movies->size(); ++j) {
	uint32_t m = (*movies)[j];
	yval_t y = _ratings.r(u,m);
	
	get_phi(u, m, phi, lsum);
	debug("phi for  %d,%d: %s", u, m, phi.s().c_str());

	if (y > 1)
	  phi.scale(y);

	_zusers.add_slice(u, phi);
	_zitems.add_slice(m, phi);
	if (_iter == 0) 
	  _user_budget[u] += y;
      }
      debug("user budget for %d = %u", u, _user_budget[u]);
    }

    // TODO: think about stochastically optimizing the \tau
    update_sticks();        // q(v_uk; tau_uk)

    update_sticks_scalar(); // q(s_u; gamma_u)
    update_items();         // q(beta_i; lambda_i)

    if (!_env.vb && !_env.natgrad_seq_updates) {
      // natgrad parallel updates
      weighted_update_all();
      compute_all_expectations();
      recompute_sums();
    }

    debug("ebeta sum = %s", _ebetasum.s().c_str());
    debug("etheta sum = %s", _ethetasum.s().c_str());

    if (_env.save_state_now) {
      lerr("Saving state at iteration %d duration %d secs", _iter, duration());
      auc();
      save_model();
      _env.save_state_now = false;
    }

    lerr("Iteration %d\n", _iter);
    compute_likelihood(false);
    compute_likelihood(true);
    
    if (_iter % _env.reportfreq == 0) {
      compute_and_write_thetas();

      printf("Iteration %d\n", _iter);
      fflush(stdout);
      
      lerr("auc..");
      auc();
      lerr("done; saving model...");
      save_model();
      lerr("done");
    }
    
    if (_iter == 0)
      write_user_budgets();

    _iter++;
    set_rho();
  }
}

void
BNPRec::set_rho()
{
  _rho = pow(_tau0 + _iter , -1 * _kappa);
}

void
BNPRec::compute_pi()
{
  double **pid = _pi.data();
  double **thetad = _theta.data();
  double **lpid = _logpi.data();
  const double *sd = _s.expected_v().const_data();
  double **vd = _v.data();
  for (uint32_t u = 0; u < _n; ++u) {
    double lw = .0;
    for (uint32_t k = 0; k < _k; ++k) {
      if (vd[u][k] < 1e-30)
	vd[u][k] = 1e-30;
      if (k > 0)
	lw += log(1 - vd[u][k-1]);
      lpid[u][k] = log(vd[u][k]) + lw;
      pid[u][k] = exp(lpid[u][k]);
      thetad[u][k] = pid[u][k] * sd[u];
    }
  }
}

void
BNPRec::compute_pi(uint32_t u)
{
  double **pid = _pi.data();
  double **thetad = _theta.data();
  double **lpid = _logpi.data();
  const double *sd = _s.expected_v().const_data();
  double **vd = _v.data();
  double lw = .0;
  for (uint32_t k = 0; k < _k; ++k) {
    if (vd[u][k] < 1e-30)
	vd[u][k] = 1e-30;
    if (k > 0)
      lw += log(1 - vd[u][k-1]);
    lpid[u][k] = log(vd[u][k]) + lw;
    pid[u][k] = exp(lpid[u][k]);
    thetad[u][k] = pid[u][k] * sd[u];
  }
}

void
BNPRec::clear_state()
{
  _zusers.zero();
  _zitems.zero();
}

//
// q(z_ui; phi_ui) multinomial updates (Eq. 5)
// 
void
BNPRec::get_phi(uint32_t n, uint32_t m, Array &phi, double &logsum) const
{
  assert (n < _n && m < _m);
  const double  **elogbeta = _beta.expected_logv().const_data();
  phi.zero();
  for (uint32_t k = 0; k < _k; ++k)
    phi[k] = elogtheta(n, k) + elogbeta[m][k];
  
  Array s(2);
  s[0] = phi.logsum();
  s[1] = compute_mult_normalizer_infsum(n);

  // Eq. 6
  logsum = s.logsum();
  phi.lognormalize(logsum);
}

// note: truncation (0,_k-1): before (_k,\infty): after

// E [log theta] prior to truncation  (Eq. 10)
double
BNPRec::elogtheta(uint32_t u, uint32_t k) const
{
  // Eq. 10
  const Array &elogsu = _s.expected_logv();
  const double **vd = _v.const_data();
  const double **logpid = _logpi.const_data();
  return elogsu[u] + logpid[u][k];
}
  
// E [log theta] at truncation  (Eq. 11)
double
BNPRec::elogtheta_at_truncation(uint32_t u) const
{
  // Eq. 11
  const Array &elogsu = _s.expected_logv();
  const double **vd = _v.const_data();
  const double **logpid = _logpi.const_data();

  double elogvt = gsl_sf_psi(1) - gsl_sf_psi(1+_alpha);
  return elogsu[u] + elogvt + \
    logpid[u][_k-1] - log(vd[u][_k-1]) + log(1 - vd[u][_k-1]);
}

double
BNPRec::elogbeta_at_truncation(uint32_t /*u*/) const
{
  return gsl_sf_psi(_beta_shape_prior) - log(_beta_rate_prior);
}

double
BNPRec::compute_X_at_truncation(uint32_t u) const
{
  return elogtheta_at_truncation(u) + elogbeta_at_truncation(u);
}

//
// The infinite part of the multinomial normalizer (Eq. 8)
//
double
BNPRec::compute_mult_normalizer_infsum(uint32_t u) const
{
  double X = compute_X_at_truncation(u);
  double elogv_t = gsl_sf_psi(_alpha) - gsl_sf_psi(1+_alpha);
  return X - log(1 - exp(elogv_t));
}


//
// q(v_uk; tau_uk)
// 
void
BNPRec::update_sticks()
{
  double **thetad = _theta.data();
  double **lpid = _logpi.data();
  debug("current sticks = %s", _v.s().c_str());
  double **vd = _v.data();
  const double **ad = _A.const_data();
  const double **zusersd = _zusers.const_data();
  const double *sd = _s.expected_v().const_data();
  double **pid = _pi.data();
  
  for (uint32_t u = 0; u < _n; ++u) {
    double lw = .0;
    for (uint32_t k = 0; k < _k; ++k) {
      //
      // \tau_{u0}...\tau_{u{k-1}} are new
      // \pi_{u0}...\pi_{u{k-1}} are new
      // lpid {u0..uk-1} are new
      //
      debug("u = %d, k = %d", u, k);
      double lpid_at_T = .0;
      double Auk = sd[u] * (-1 * prod_at_k(u,k) / vd[u][k]
			    + sum_of_prod_in_range(u, k+1, lpid_at_T)/(1 - vd[u][k]) 
			    + compute_scalar_rate_infsum(u, lpid_at_T)/(1 - vd[u][k]));
      if (u == 1)
	debug("u = %d, k = %d, Auk = %.10f, Buk = %.10f, Cuk = %.10f", 
	      u, k, Auk, B(u,k,Auk), -zusersd[u][k]);

      if (fabs(Auk) < 1e-30) {
	double x = zusersd[u][k] / B(u,k,Auk);
	if (x < 1e-30)
	  vd[u][k] = 1e-30;
	else 
	  vd[u][k] = x;
      } else
	vd[u][k] = solve_quadratic(Auk, B(u,k,Auk), -zusersd[u][k]);

      if (k > 0)
	pid[u][k] = (pid[u][k-1] / vd[u][k-1]) * (1 - vd[u][k-1]) * vd[u][k];
      else
	pid[u][k] = vd[u][k];

      lpid[u][k] = log(pid[u][k]);
      thetad[u][k] = pid[u][k] * sd[u];
    }
  }
  debug("sticks = %s", _v.s().c_str());
}

double
BNPRec::solve_quadratic(double a, double b, double c)
{
  double s1 = (-b + sqrt(b*b - 4*a*c)) / (2*a);
  double s2 = (-b - sqrt(b*b - 4*a*c)) / (2*a);
  debug("A = %f, B = %f, C = %f", a, b, c);
  debug("s1 = %f, s2 = %f", s1, s2);

  if (s1 > .0 && s1 <= 1.0 && s2 > .0 && s2 <= 1.0) {
    lerr("s1 %f and s2 %f are out of range in solve_quadratic()", s1, s2);
    lerr("a = %.5f, b = %.5f, c = %.5f\n", a, b, c);

    if (s1 < s2)
      return s1 + 1e-30;
    else
      return s2 + 1e-30;
  }

  if (s1 > .0 && s1 <= 1.0)  
    return s1;

  if (s2 > .0 && s2 <= 1.0)
    return s2;
  
  // TODO
  if (fabs(s1 - .0) < 1e-30)
    return 1e-30;
    
  if (fabs(s1 - 1.0) < 1e-30)
    return 1 - 1e-30;

  if (fabs(s2 - .0) < 1e-30)
    return 1e-30;
    
  if (fabs(s2 - 1.0) < 1e-30)
    return 1 - 1e-30;

  lerr("WARNING: s1 %.10f and s2 %.10f are out of range in solve_quadratic()", s1, s2);
  assert(0);
  return s1;
}

void
BNPRec::recompute_ebetasum()
{
  const double **ebeta = _beta.expected_v().const_data();
  _ebetasum.zero();
  for (uint32_t k = 0; k < _k; ++k)
    for (uint32_t m = 0; m < _m; ++m)
      _ebetasum[k] += ebeta[m][k];
}

double
BNPRec::sum_of_prod_in_range(uint32_t u, uint32_t K, double &pid_at_T)
{
  double **pid = _pi.data();
  double sum =  .0;
  double p = convert_oldpi_to_new(u,K-1); // pi_{uk}
  for (uint32_t k = K; k < _k; ++k) {
    p = convert_oldpi_to_new(p,u,k);
    sum += p * _ebetasum[k];
  }
  pid_at_T = p;
  return sum;
}

double
BNPRec::convert_oldpi_to_new(double pi_at_kminus1, 
			     uint32_t u, uint32_t k) const
{
  //
  // pi_at_kminus1 is current (new)
  // 
  const double **vd = _v.const_data();
  return pi_at_kminus1 * ((1 - vd[u][k-1]) / vd[u][k-1]) * vd[u][k];
}

double
BNPRec::convert_oldpi_to_new(uint32_t u, uint32_t k) const
{
  //
  // valid only if lpid[u][k-1] is current (new)
  //
  const double **pid = _pi.const_data();
  const double **vd = _v.const_data();
  if (k == 0)
    return pid[u][0];
  assert(k > 0);
  return pid[u][k-1] * ((1 - vd[u][k-1])  / vd[u][k-1]) * vd[u][k];
}

double
BNPRec::prod_at_k(uint32_t u, uint32_t k)
{
  return convert_oldpi_to_new(u, k) * _ebetasum[k];
}

double
BNPRec::B(uint32_t u, uint32_t k, double Auk)
{
  const double **zusersd = _zusers.const_data();
  double **ad = _A.data();
  return _alpha - 1 + zusersd[u][k] -		\
    Auk + compute_zusers_sum(u, k);
}

double
BNPRec::compute_zusers_sum(uint32_t u, uint32_t tok)
{
  const double **zusersd = _zusers.const_data();  
  double sum = .0;
  for (uint32_t k = 0; k <= tok; ++k)
    sum += zusersd[u][k];
  return (double)_user_budget[u] - sum;
}

 
double
BNPRec::compute_zusers_sum2(uint32_t u, uint32_t fromk)
{
  const double **zusersd = _zusers.const_data();  
  double sum = .0;
  for (uint32_t k = fromk; k < _k; ++k)
    sum += zusersd[u][k];
  return sum;
}

//
// q(s_u; gamma_u)
// 
void
BNPRec::update_sticks_scalar()
{
  const double **zusersd = _zusers.const_data();
  
  for (uint32_t u = 0; u < _n; ++u) {
    double infsum = compute_scalar_rate_infsum(u);
    double fnsum = compute_scalar_rate_finitesum(u);
    debug("updating s shape of %d with %f\n", u, _user_budget[u]);
    _s.update_shape_next(u, _user_budget[u]);
    debug("updating s rate of %d with %f:%f\n", u, fnsum, infsum);
    _s.update_rate_next(u, fnsum + infsum);
  }
  if (_env.vb) {
    _s.swap();
    _s.compute_expectations();
    recompute_ethetasum();
  } else if (_env.natgrad_seq_updates) {
    _s.weighted_update(_rho);
    _s.compute_expectations();
    recompute_ethetasum();
  }
}

double
BNPRec::compute_scalar_rate_infsum(uint32_t u) const
{
  double Y = compute_Y(u);
  double D = (_beta_shape_prior / _beta_rate_prior) * _m;
  return Y * D;
}

double
BNPRec::compute_scalar_rate_infsum(uint32_t u, double pid_at_T) const
{
  const double **vd = _v.const_data();  
  double Y = (pid_at_T / vd[u][_k-1]) * (1 - vd[u][_k-1]);
  double D = (_beta_shape_prior / _beta_rate_prior) * _m;
  return Y * D;
}

double
BNPRec::compute_Y(uint32_t u) const
{
  const double **vd = _v.const_data();  
  const double **pid = _pi.const_data();
  const double **lpid = _logpi.const_data();
  //return exp(lpid[u][_k-1] - log(vd[u][_k-1]) + log(1 - vd[u][_k-1]));
  return (pid[u][_k-1] / vd[u][_k-1]) * (1 - vd[u][_k-1]);
}

double
BNPRec::compute_scalar_rate_finitesum(uint32_t u) const
{
  double sum = .0;
  const double **vd = _v.const_data();  
  const double **pid = _pi.const_data();
  for (uint32_t k = 0; k < _k; ++k)
    sum += pid[u][k] * _ebetasum[k];
  return sum;
}

//
// q(beta_i; lambda_i)
//

void
BNPRec::update_items()
{
  const double **zitemsd = _zitems.const_data();
  Array phi(_k);
  for (uint32_t m = 0; m < _m; ++m) {
    for (uint32_t k = 0; k < _k; ++k) // optimize
      phi[k] = zitemsd[m][k];
    _beta.update_shape_next(m, phi);
  }
  _beta.update_rate_next(_ethetasum);

  if (_env.vb) {
    _beta.swap();
    _beta.compute_expectations();
    recompute_ebetasum();
  } else if (_env.natgrad_seq_updates) {
    _beta.weighted_update(_rho);
    _beta.compute_expectations();
    recompute_ebetasum();
  }
}

void
BNPRec::recompute_ethetasum()
{
  const double **pid = _pi.const_data();
  const double *sd = _s.expected_v().const_data();
  double *ethetasumd =  _ethetasum.data();

  _ethetasum.zero();
  for (uint32_t u = 0; u < _n; ++u)
    for (uint32_t k = 0; k < _k; ++k)
      _ethetasum[k] += sd[u] * pid[u][k];
}

void
BNPRec::swap_all()
{
  // the sticks don't need swapping
  _s.swap();
  _beta.swap();
}

void
BNPRec::weighted_update_all()
{
  _s.weighted_update(_rho);
  _beta.weighted_update(_rho);
  // the sticks don't need updating
}

void
BNPRec::compute_all_expectations()
{
  // the sticks don't require expectations
  _s.compute_expectations();
  _beta.compute_expectations();
}

void
BNPRec::recompute_sums()
{
  recompute_ethetasum();
  recompute_ebetasum();
}

void
BNPRec::save_model()
{
  _v.save(Env::file_str("/sticks").c_str(), _ratings.seq2user());
  _beta.save(_ratings.seq2movie());
  _s.save(_ratings.seq2user());
  _theta.save(Env::file_str("/theta.tsv").c_str(), _ratings.seq2user());
}

void
BNPRec::write_user_budgets()
{
  FILE *f = fopen(Env::file_str("/budgets.tsv").c_str(), "w");
  assert(f);
  for (uint32_t i = 0; i < _n; ++i) {
    const IDMap &m = _ratings.seq2user();
    IDMap::const_iterator idt = m.find(i);
    if (idt != m.end()) {
      fprintf(f,"%d\t", i);
      fprintf(f,"%d\t", (*idt).second);
      fprintf(f,"%d\n", _user_budget[i]);
    }
  }
  fclose(f);
}

double
BNPRec::compute_sum_theta_beta() const
{
  double s = .0;
  const double *sd = _s.expected_v().const_data();
  for (uint32_t u = 0; u < _n; ++u) {
    double infsum = compute_scalar_rate_infsum(u);
    double fnsum = compute_scalar_rate_finitesum(u);
    s += (infsum + fnsum) * sd[u];
  }
  return s;
}


//
// ELBO // todo: add log(y_{ui}!)
//
void
BNPRec::approx_log_likelihood() const
{
  const double **vd = _v.const_data();
  const double **ebeta = _beta.expected_v().const_data();
  const double **elogbeta = _beta.expected_logv().const_data();
  const double *sd = _s.expected_v().const_data();
  const double *slogd = _s.expected_logv().const_data();

  double s = .0;
  double s1 = .0, s2 = .0, s3 = .0, s4 = .0;
  Array phi(_k);
  double lsum = .0;
  
  for (uint32_t u = 0; u < _n; ++u) {
    const vector<uint32_t> *movies = _ratings.get_movies(u);      
    
    for (uint32_t j = 0; j < movies->size(); ++j) {
      uint32_t m = (*movies)[j];
      yval_t y = _ratings.r(u,m);
      
      get_phi(u, m, phi, lsum);
      s += y * lsum;
    }
  }
  s1 = s;

  // prior v_uk
  for (uint32_t u = 0; u < _n; ++u)
    for (uint32_t k = 0; k < _k; ++k)
      s += log (_alpha) + (_alpha - 1) * log (1 - vd[u][k]);
  s2 = s - s1;


  s -= compute_sum_theta_beta();
  s3 = s - s2;

  // gamma variables: prior and posterior terms
  s += _beta.compute_elbo_term();
  s += _s.compute_elbo_term();
  s4 = s - s3;
  
  fprintf(_af, "%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n", s1, s2, s3, s4, s);
  fflush(_af);
}

void
BNPRec::load_validation_and_test_sets()
{
  char buf[4096];
  sprintf(buf, "%s/validation.tsv", _env.datfname.c_str());
  FILE *validf = fopen(buf, "r");
  assert(validf);
  _ratings.read_generic(validf, &_validation_map);
  fclose(validf);

  sprintf(buf, "%s/test.tsv", _env.datfname.c_str());
  FILE *testf = fopen(buf, "r");
  assert(testf);
  _ratings.read_generic(testf, &_test_map);
  fclose(testf);
  printf("+ loaded validation and test sets from %s\n", _env.datfname.c_str());
  fflush(stdout);
  Env::plog("test ratings", _test_map.size());
  Env::plog("validation ratings", _validation_map.size());
}

void
BNPRec::compute_likelihood(bool validation)
{
  uint32_t k = 0, kzeros = 0, kones = 0;
  double s = .0, szeros = 0, sones = 0;
  
  CountMap *mp = NULL;
  FILE *ff = NULL;
  if (validation) {
    mp = &_validation_map;
    ff = _vf;
  } else {
    mp = &_test_map;
    ff = _tf;
  }

  for (CountMap::const_iterator i = mp->begin();
       i != mp->end(); ++i) {
    const Rating &e = i->first;
    uint32_t n = e.first;
    uint32_t m = e.second;

    yval_t r = i->second;
    double u = pair_likelihood(n,m,r);
    s += u;
    k += 1;
  }
  info("s = %.5f\n", s);
  fprintf(ff, "%d\t%d\t%.9f\t%d\n", _iter, duration(), s / k, k);
  fflush(ff);
  double a = s / k;

  if (!validation)
    return;

  bool stop = false;
  int why = -1;
  if (_iter > 10) {
    if (a > _prev_h && _prev_h != 0 && fabs((a - _prev_h) / _prev_h) < 0.00001) {
      stop = true;
      why = 0;
    } else if (a < _prev_h)
      _nh++;
    else if (a > _prev_h)
      _nh = 0;

    if (_nh > 1) { // be robust to small fluctuations in predictive likelihood
      why = 1;
      stop = true;
    }
  }
  _prev_h = a;
  FILE *f = fopen(Env::file_str("/max.tsv").c_str(), "w");
  fprintf(f, "%d\t%d\t%.5f\t%d\n", 
	  _iter, duration(), a, why);
  fclose(f);
  if (stop) {
    do_on_stop();
    exit(0);
  }
}

void
BNPRec::do_on_stop()
{
  compute_and_write_thetas();
  save_model();
  _save_ranking_file = true;
  auc();
  _save_ranking_file = false;
}



uint32_t
BNPRec::factorial(uint32_t n)  const
{ 
  //return n <= 1 ? 1 : (n * factorial(n-1));
  uint32_t v = 1;
  for (uint32_t i = 2; i <= n; ++i)
    v *= i;
  return v;
} 

double
BNPRec::pair_likelihood(uint32_t p, uint32_t q, yval_t y) const
{
  const double **vd = _v.const_data();
  const double *sd = _s.expected_v().const_data();
  const double **ebeta = _beta.expected_v().const_data();
  double s = .0;
  double w = 1.0;
  for (uint32_t k = 0; k < _k; ++k) {
    if (k > 0)
      w *= (1 - vd[p][k-1]);
    s += sd[p] * vd[p][k] * w * ebeta[q][k];
  }
  // infinite sum
  s += sd[p] * compute_Y(p) * _beta_shape_prior / _beta_rate_prior;
  
  if (s < 1e-30)
    s = 1e-30;
  info("%d, %d, s = %f, f(y) = %ld\n", p, q, s, factorial(y));
  return y * log(s) - s - log(factorial(y));
}

void
BNPRec::auc()
{
  double mhits10 = 0, mhits100 = 0;
  uint32_t total_users = 0;
  FILE *f = 0;
  if (_save_ranking_file)
    f = fopen(Env::file_str("/ranking.tsv").c_str(), "w");
  
  if (!_save_ranking_file) {
    _sampled_users.clear();
    do {
      uint32_t n = gsl_rng_uniform_int(_r, _n);
      _sampled_users[n] = true;
    } while (_sampled_users.size() < 1000 && _sampled_users.size() < _n / 2);
  }
  
  KVArray mlist(_m);
  for (UserMap::const_iterator itr = _sampled_users.begin();
       itr != _sampled_users.end(); ++itr) {
    uint32_t n = itr->first;
    
    for (uint32_t m = 0; m < _m; ++m) {
      if (_ratings.r(n,m) > 0) { // skip training
	mlist[m].first = m;
	mlist[m].second = .0;
	continue;
      }
      double u = link_prob(n,m);
      mlist[m].first = m;
      mlist[m].second = u;
    }
    uint32_t hits10 = 0, hits100 = 0;
    mlist.sort_by_value();
    for (uint32_t j = 0; j < mlist.size() && j < 100; ++j) {
      KV &kv = mlist[j];
      uint32_t m = kv.first;
      double pred = kv.second;
      Rating r(n, m);

      uint32_t m2 = 0, n2 = 0;
      if (_save_ranking_file) {
	IDMap::const_iterator it = _ratings.seq2user().find(n);
	assert (it != _ratings.seq2user().end());
	
	IDMap::const_iterator mt = _ratings.seq2movie().find(m);
	if (mt == _ratings.seq2movie().end())
	  continue;
      
	m2 = mt->second;
	n2 = it->second;
      }

      CountMap::const_iterator itr = _test_map.find(r);
      if (itr != _test_map.end()) {
	int v = itr->second;
	v = _ratings.rating_class(v);
	assert(v > 0);
	if (_save_ranking_file) {
	  if (_ratings.r(n, m) == .0) // skip training
	    fprintf(f, "%d\t%d\t%.5f\t%d\n", n2, m2, pred, v);
	}
	
	if (j < 10) {
	  hits10++;
	  hits100++;
	} else if (j < 100) {
	  hits100++;
	}
      } else {
	if (_save_ranking_file) {
	  if (_ratings.r(n, m) == .0) // skip training
	    fprintf(f, "%d\t%d\t%.5f\t%d\n", n2, m2, pred, 0);
	}
      }
    }
    mhits10 += (double)hits10 / 10;
    mhits100 += (double)hits100 / 100;
    total_users++;
  }
  if (_save_ranking_file)
    fclose(f);
  lerr("sampled users size = %d, total_users = %d", _sampled_users.size(), total_users);
  fprintf(_pf, "%.5f\t%.5f\n", 
	  (double)mhits10 / total_users, 
	  (double)mhits100 / total_users);
  fflush(_pf);
}


double
BNPRec::link_prob(uint32_t p, uint32_t q) const
{
  const double **vd = _v.const_data();
  const double *sd = _s.expected_v().const_data();
  const double **ebeta = _beta.expected_v().const_data();
  double s = .0;
  double w = 1.0;
  for (uint32_t k = 0; k < _k; ++k) {
    if (k > 0)
      w *= (1 - vd[p][k-1]);
    s += sd[p] * vd[p][k] * w * ebeta[q][k];
  }
  // infinite sum
  s += sd[p] * compute_Y(p) * _beta_shape_prior / _beta_rate_prior;
  
  if (s < 1e-30)
    s = 1e-30;
  double prob_zero = exp(-s);
  return 1 - prob_zero;
}

void
BNPRec::compute_top_factors()
{
  const double *sd = _s.expected_v().const_data();
  double **thetad = _theta.data();
  uArray used(_k);
  used.zero();

  Array sum(_k);
  for (uint32_t u = 0; u < _n; ++u)  {
    sum.zero();
    double fsum = .0, isum = .0;

    // finite sum
    for (uint32_t k = 0; k < _k; ++k)  {
      sum[k] += thetad[u][k] * _ebetasum[k];
      fsum += sum[k];
    }

    // infinite sum
    double D = (_beta_shape_prior / _beta_rate_prior) * _m;
    isum = sd[u] * compute_Y(u) * D;

    double S = fsum + isum;

    KVArray kvlist(_k);
    for (uint32_t k = 0; k < _k; ++k)
      kvlist[k] = KV(k, sum[k]);
    kvlist.sort_by_value();

    double s = .0;
    vector<uint32_t> factors;
    for (uint32_t i = 0; i < kvlist.size(); ++i) {
      s += kvlist[i].second;
      if ((s + isum) / S > 0.9)
	break;
      used[kvlist[i].first]++;
    }
  }
  IDMap idmap;
  used.save(Env::file_str("/used_factors.tsv"), idmap);

  uint32_t nused = 0;
  for (uint32_t k = 0; k < _k; ++k)
    if (used[k] > 0)
      nused++;
  Env::plog("number of factors used:", nused);
}

void
BNPRec::compute_and_write_thetas()
{
  _active_k.zero();
  uint32_t active_factors = 0;
  const double *sd = _s.expected_v().const_data();
  const double **thetad = _theta.const_data();
  uint32_t k = 0;
  double epsilon = 1e-5;
  do {
    for (uint32_t u = 0; u < _n; ++u)  {
      if (thetad[u][k] > epsilon) {
	_active_k[k] = true;
	break;
      }
    }
    k++;
  } while (k < _k);
  FILE *f = fopen(Env::file_str("/active_k.tsv").c_str(), "w");
  for (uint32_t k = 0; k < _k; ++k) {
    if (_active_k[k])
      active_factors++;
    fprintf(f, "%d\n", _active_k[k]);
  }
  fclose(f);
  lerr("active factors = %d", active_factors);
}

void
BNPRec::gen_ranking_for_users()
{
  load_validation_and_test_sets();
  load_model();

  char buf[4096];
  sprintf(buf, "%s/test_users.tsv", _env.datfname.c_str());
  FILE *f = fopen(buf, "r");
  assert(f);
  _ratings.read_test_users(f, &_sampled_users);
  fclose(f);

  compute_top_factors();

  _save_ranking_file = true;
  auc();
  _save_ranking_file = false;
  printf("DONE writing ranking.tsv in output directory\n");
  fflush(stdout);
}

void
BNPRec::load_model()
{
  _v.load("sticks");
  _beta.load();
  _s.load();
  compute_pi();
  recompute_sums();
}
