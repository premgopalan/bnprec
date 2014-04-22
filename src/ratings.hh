#ifndef RATINGS_HH
#define RATINGS_HH

#include <string>
#include <vector>
#include <queue>
#include <map>
#include <stdint.h>
#include "matrix.hh"
#include "env.hh"
#include <string.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf.h>

using namespace std;

class Ratings {
public:
  Ratings(Env &env):
    _users2rating(env.n),
    _users(env.n),
    _movies(env.m),
    _env(env),
    _curr_user_seq(0), 
    _curr_movie_seq(0),
    _nratings(0),
    _likes(0) { }
  ~Ratings() { }

  int read(string s);
  const uint8_t rating_class(uint32_t v);
  
  const SparseMatrix &users() const { return _users; }
  SparseMatrix &users() { return _users; }

  const SparseMatrix &movies() const { return _movies; }
  SparseMatrix &movies() { return _movies; }
  
  uint32_t n() const;
  uint32_t m() const;
  uint32_t r(uint32_t i, uint32_t j) const;
  uint32_t nratings() const { return _nratings; }
  uint32_t likes() const { return _likes; }
  const vector<Rating> &allratings() const { return _ratings; }
 
  const vector<uint32_t> *get_users(uint32_t a);
  const vector<uint32_t> *get_movies(uint32_t a);

  const IDMap &user2seq() const { return _user2seq; }
  const IDMap &seq2user() const { return _seq2user; }

  const IDMap &movie2seq() const { return _movie2seq; }
  const IDMap &seq2movie() const { return _seq2movie; }
  int read_generic(FILE *f, CountMap *m);
  void load_movies_metadata(string dir);
  int read_test_users(FILE *f, UserMap *);

  string movie_type(uint32_t movie_seq) const;
  string movie_name(uint32_t movie_seq) const;
  
private:
  int read_generic_train(string dir);
  int read_netflix_movie(string dir, uint32_t movie);
  int read_movielens(string dir);
  int read_mendeley(string dir);
  int read_echonest(string dir);
  int read_movielens_metadata(string dir);
  int read_netflix_metadata(string dir);
  int read_mendeley_metadata(string dir);
  string movies_by_user_s() const;
  bool add_movie(uint32_t id);
  bool add_user(uint32_t id);

  SparseMatrixR _users2rating;
  SparseMatrix _users;
  SparseMatrix _movies;
  vector<Rating> _ratings;

  Env &_env;
  IDMap _user2seq;
  IDMap _movie2seq;
  IDMap _seq2user;
  IDMap _seq2movie;
  StrMap _str2id;
  StrMapInv _user2str;
  StrMapInv _movie2str;
  uint32_t _curr_user_seq;
  uint32_t _curr_movie_seq;
  uint32_t _nratings;
  uint32_t _likes;
  StrMapInv _movie_names;
  StrMapInv _movie_types;
};

inline uint32_t
Ratings::n() const
{
  return _users.size();
}

inline uint32_t
Ratings::m() const
{
  return _movies.size();
}

inline bool
Ratings::add_user(uint32_t id)
{
  if (_curr_user_seq >= _env.n) {
    lerr("max users %d reached", _env.n);
    return false;
  }
  _user2seq[id] = _curr_user_seq;
  _seq2user[_curr_user_seq] = id;

  assert (!_users[_curr_user_seq]);
  std::vector<uint32_t> **v = _users.data();
  v[_curr_user_seq] = new vector<uint32_t>;
  RatingMap **rm = _users2rating.data();
  rm[_curr_user_seq] = new RatingMap;
  _curr_user_seq++;
  return true;
}

inline bool
Ratings::add_movie(uint32_t id)
{
  if (_curr_movie_seq >= _env.m) {
    //lerr("max movies %d reached", _env.m);
    return false;
  }
  _movie2seq[id] = _curr_movie_seq;
  _seq2movie[_curr_movie_seq] = id;

  assert (!_movies[_curr_movie_seq]);
  std::vector<uint32_t> **v = _movies.data();
  v[_curr_movie_seq] = new vector<uint32_t>;
  _curr_movie_seq++;
  return true;
}

inline uint32_t
Ratings::r(uint32_t a, uint32_t b) const
{
  assert (a < _env.n && b < _env.m);
  const RatingMap *rm = _users2rating[a];
  if (!rm) {
    IDMap::const_iterator itr = _seq2user.find(a);
    if (itr == _seq2user.end())
      lerr("cannot find ID for seq %d (item %d)", a, b);
    else
      lerr("a = %d (%d), b = %d, ", a,  itr->second, b);
  }
  assert(rm);
  const RatingMap &rmc = *rm;
  RatingMap::const_iterator itr = rmc.find(b);
  return itr->second;
}

inline const vector<uint32_t> *
Ratings::get_users(uint32_t a)
{
  assert (a < _movies.size());
  const vector<uint32_t> *v = _movies[a];
  return v;
}

inline const vector<uint32_t> *
Ratings::get_movies(uint32_t a)
{
  assert (a < _users.size());
  const vector<uint32_t> *v = _users[a];
  return v;
}

inline const uint8_t
Ratings::rating_class(uint32_t v)
{
  return v >= 1  ? 1 : 0;
}

inline string
Ratings::movie_name(uint32_t movie_seq) const
{
  assert (movie_seq < _env.m);
  StrMapInv::const_iterator i = _movie_names.find(movie_seq);
  if (i != _movie_names.end())
    return i->second;
  return "";
}

inline string
Ratings::movie_type(uint32_t movie_seq) const
{
  assert (movie_seq < _env.m);
  StrMapInv::const_iterator i = _movie_types.find(movie_seq);
  if (i != _movie_types.end())
    return i->second;
  return "";
}

#endif
