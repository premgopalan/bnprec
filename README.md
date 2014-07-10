Installation
------------

Required libraries: gsl, gslblas, pthread

On Linux/Unix run

 ./configure
 make; make install

On Mac OS, the location of the required gsl, gslblas and pthread
libraries may need to be specified:

 ./configure LDFLAGS="-L/opt/local/lib" CPPFLAGS="-I/opt/local/include"
 make; make install

The binary 'gaprec' will be installed in /usr/local/bin unless a
different prefix is provided to configure. (See INSTALL.)

GAPREC: Gamma Poisson factorization based recommendation tool
--------------------------------------------------------------

**gaprec** [OPTIONS]

   -dir <string>    path to dataset directory with 3 files:
   		    train.tsv, test.tsv, validation.tsv
		    (for examples, see example/movielens-1m)
 
   -m <int>	  number of items
   -n <int>	  number of users
   -T <int>	  truncation level
   
   -rfreq <int>	  assess convergence and compute other stats 
   		  <int> number of iterations
		  default: 10

   -alpha         set Gamma shape hyperparameter alpha (see paper)
   -C             set Gamma scale hyperparameter c     (see paper)

   -label         add a tag to the output directory

   -gen-ranking	  generate ranking file to use in precision 
   		  computation; see example		  


Example
--------

(1) ../src/gaprec -dir ../example/movielens -n 6040 -m 3900  -T 100 -rfreq 10 

This will write output in n6040-m3900-k100-batch-alpha1.1-scale1-vb

You can change the settings for the Gamma hyperparameters alpha and c (see paper) using the -alpha and the -C options.

To generate the ranking file (ranking.tsv) for precision computation, run the following:

(2) cd n6040-m3900-k100-batch-alpha1.1-scale1-vb

../../src/gaprec -dir ../../example/movielens -n 6040 -m 3900  -T 100 -rfreq 10 -gen-ranking

This will rank all y == 0 in training and the test.tsv pairs in
decreasing order of their scores, along with the non-zero ratings from
test.tsv.

The output is now in a new directory within the fit. Look for ranking.tsv.

