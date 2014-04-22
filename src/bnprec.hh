#ifndef BNPREC_HH
#define BNPREC_HH

#include "env.hh"
#include "ratings.hh"
#include "gammapoisson.hh"

class BNPRec {
public:
  BNPRec(Env &env, Ratings &ratings);
  ~BNPRec();
  
  void batch_infer();
  void gen_ranking_for_users();


private:
  void initialize();
  void approx_log_likelihood() const;
  void compute_and_write_thetas();

  void initialize_sticks();
  void swap_all();
  void weighted_update_all();
  void compute_all_expectations();
  void save_model();
  void load_model();

  void get_phi(uint32_t n, uint32_t m, Array &phi, double &logsum) const;
  void save_user_state(string s, const Matrix &mat);
  void save_item_state(string s, const Matrix &mat);
  void save_state(string s, const Array &mat);

  void load_validation_and_test_sets();
  void compute_likelihood(bool validation);
  void write_user_budgets();

  double elogtheta(uint32_t u, uint32_t k) const;
  double elogtheta_at_truncation(uint32_t u) const;
  double elogbeta_at_truncation(uint32_t u) const; 
  double compute_X_at_truncation(uint32_t u) const;
  double compute_mult_normalizer_infsum(uint32_t u) const;
  double compute_sum_theta_beta() const;
  
  void update_sticks();
  void update_sticks_scalar();
  
  void set_rho();

  void recompute_ebetasum();
  void recompute_A();
  void compute_pi();
  void compute_pi(uint32_t u);
  double B(uint32_t u, uint32_t k, double Auk);
  double solve_quadratic(double a, double b, double c);
  double sum_of_prod_in_range(uint32_t u, uint32_t K, double &lpid_at_T);
  double prod_at_k(uint32_t u, uint32_t k);
  double compute_zusers_sum(uint32_t u, uint32_t fromk);
  double compute_zusers_sum2(uint32_t u, uint32_t fromk);
  double convert_oldpi_to_new(double logpi_at_kminus1, 
				 uint32_t u, uint32_t k) const;
  double convert_oldpi_to_new(uint32_t u, uint32_t k) const;

  double compute_scalar_rate_infsum(uint32_t u) const;
  double compute_scalar_rate_finitesum(uint32_t u) const;
  double compute_scalar_rate_infsum(uint32_t u, double lpid_at_T) const;
  double compute_Y(uint32_t u) const;

  void update_items();
  void recompute_ethetasum();

  void clear_state();
  void recompute_sums();
  uint32_t duration() const;
  double pair_likelihood(uint32_t p, uint32_t q, yval_t y) const;
  void do_on_stop();
  uint32_t factorial(uint32_t n)  const;
  double link_prob(uint32_t user, uint32_t movie) const;
  void auc();
  void compute_top_factors();

  Env &_env;
  Ratings &_ratings;

  uint32_t _n;
  uint32_t _m;

  uint32_t _k;      // truncation level
  uint32_t _iter;

  double _c;
  double _alpha;

  double _beta_shape_prior;
  double _beta_rate_prior;

  GPMatrixGR _beta; // \beta for all items
  GPArray _s;       // s_u for all users
  Matrix _v;        // variational sticks

  Matrix _zusers;
  Matrix _zitems;
  Array _user_budget;
  Array _ebetasum;
  Array _ethetasum;
  Matrix _A;
  Array _scalar_finite_sums;
  Matrix _pi;
  Matrix _theta;
  Matrix _logpi;

  uArray _active_k;

  uint32_t _start_time;
  gsl_rng *_r;
  FILE *_af;
  FILE *_vf;
  FILE *_tf;
  FILE *_pf;

  uint32_t _nh;
  double _prev_h;  

  CountMap _test_map;
  CountMap _validation_map;

  double _tau0;
  double _rho;
  double _kappa;

  bool _save_ranking_file;  
  UserMap _sampled_users;
};

inline uint32_t
BNPRec::duration() const
{
  time_t t = time(0);
  return t - _start_time;
}


#endif
