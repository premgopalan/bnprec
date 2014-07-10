#!/usr/bin/perl

use strict;
use warnings;
use Getopt::Long;

my $finbin = "/disk/scratch1/prem/gaprec/src/gaprec";
my $infbin = "/disk/scratch1/prem/bnppf/src/gaprec";
my $batch_rfreq = 1;
our $F;

my %nf = ();
$nf{loc} = "/disk/scratch1/prem/bnppf/data/netflix";
$nf{N} = 480189;
$nf{M} = 17770;
$nf{fincmd} = "$finbin  -dir $nf{loc} -m $nf{M} -n $nf{N} -k %d -rfreq $batch_rfreq -a %.3f -b %.3f -label %s";
$nf{infcmd} = "$infbin  -dir $nf{loc} -m $nf{M} -n $nf{N} -k %d -bnp -rfreq $batch_rfreq -alpha %f -scale %f -label %s";
$nf{infrankcmd} = "$infbin  -dir $nf{loc} -m $nf{M} -n $nf{N} -k %d -bnp -gen-ranking";
$nf{finrankcmd} = "$finbin  -dir $nf{loc} -m $nf{M} -n $nf{N} -k %d -gen-ranking";
$nf{mbudget} = 595.81;

my %movl = ();
$movl{loc} = "/disk/scratch1/prem/bnppf/data/movielens";
$movl{N} = 69878;
$movl{M} = 10677;
$movl{fincmd} = "$finbin  -dir $movl{loc} -m $movl{M} -n $movl{N} -k %d -rfreq $batch_rfreq -a %.3f -b %.3f -label %s";
$movl{infcmd} = "$infbin  -dir $movl{loc} -m $movl{M} -n $movl{N} -k %d -bnp -rfreq $batch_rfreq -alpha %f -scale %f -label %s";
$movl{infrankcmd} = "$infbin  -dir $movl{loc} -m $movl{M} -n $movl{N} -k %d -bnp -gen-ranking";
$movl{finrankcmd} = "$finbin  -dir $movl{loc} -m $movl{M} -n $movl{N} -k %d -gen-ranking";

my %ml = ();
$ml{loc} = "/disk/scratch1/prem/bnppf/data/movielens-1m";
$ml{N} = 6040;
$ml{M} = 3952;
$ml{fincmd} = "$finbin  -dir $ml{loc} -m $ml{M} -n $ml{N} -k %d -rfreq $batch_rfreq -a %.3f -b %.3f -label %s";
$ml{infcmd} = "$infbin  -dir $ml{loc} -m $ml{M} -n $ml{N} -k %d -bnp -rfreq $batch_rfreq -alpha %f -scale %f -label %s";
$ml{infrankcmd} = "$infbin  -dir $ml{loc} -m $ml{M} -n $ml{N} -k %d -bnp -gen-ranking";
$ml{finrankcmd} = "$finbin  -dir $ml{loc} -m $ml{M} -n $ml{N} -k %d -gen-ranking";

my %dy = ();
$dy{loc} = "/disk/scratch1/prem/bnppf/data/mendeley";
$dy{N} = 80278;
$dy{M} = 261248;
$dy{fincmd} = "$finbin  -dir $dy{loc} -m $dy{M} -n $dy{N} -k %d -rfreq $batch_rfreq -a %.3f -b %.3f  -label %s";
$dy{infcmd} = "$infbin  -dir $dy{loc} -m $dy{M} -n $dy{N} -k %d -bnp -rfreq $batch_rfreq -alpha %f -scale %f -label %s";
$dy{infrankcmd} = "$infbin  -dir $dy{loc} -m $dy{M} -n $dy{N} -k %d -bnp -gen-ranking";
$dy{finrankcmd} = "$finbin  -dir $dy{loc} -m $dy{M} -n $dy{N} -k %d -gen-ranking";
$dy{mbudget} = 50.39;

my $seed = 0;
my $K = 100;
my $bgfalse = 0;

sub run($) {
    my $a = shift @_;
    print $F "CMD = $a\n";
    if (!$bgfalse) {
	if (system("$a 2>&1 > /dev/null &") != 0) { 
	    print $F "$a failed\n";
	    return -1;
	}
    } else {
	if (system("$a 2>&1 > /dev/null") != 0) { 
	    print $F "$a failed\n";
	    return -1;
	}
    }
    return 0;
}

sub main()
{
    my $finite = 0;
    my $dataset = "netflix";
    my $label = "x";
    my $genrank = 0;
    
    GetOptions (
	'K=i' => \$K,
	'dataset=s' => \$dataset,
	'seed=i' => \$seed,
	'finite' => \$finite,
	'label=s' => \$label,
	'genrank' => \$genrank,
	'bgfalse' => \$bgfalse,
	);

    open $F, ">>cmds.txt";

    my $m = \%nf;
    if ($dataset eq "movielens-1m") {
	$m = \%ml;
    } elsif ($dataset eq "movielens") {
	$m = \%movl;
    } elsif ($dataset eq "mendeley") {
	$m = \%dy;
    }

    if ($genrank) {
	open F, "<param.txt";
	my $T = 0;
	while (<F>) {
	    if ($_ =~ /k: (\d+)/) {
		$T = $1 + 0;
		goto OUT;
	    }
	}
      OUT:
	close F;
	if ($finite) {
	    my $cmd = sprintf $m->{finrankcmd}, $T;
	    run($cmd);
	} else {
	    my $cmd = sprintf $m->{infrankcmd}, $T;
	    run($cmd);
	}
    } elsif (!$finite) {
	foreach my $C (1, 10) {
	    foreach my $T (200) {
		foreach my $alpha (1.1, 100) {
		    my $cmd = sprintf $m->{infcmd}, $T, $alpha, $C, $label;
		    run($cmd);
		}
	    }
	}
    } else {
	my $a = 1;
	my $b = 1;
	foreach my $k (10, 25, 50, 100, 150, 200) {
	    my $cmd = sprintf $m->{fincmd}, $k, $a, $b, $label;
	    run($cmd);
	}
    }

    close $F;
}


main();


#my $alpha = $m->{mbudget};
#foreach my $alpha (1.734, 1.1) {
#   foreach my $C (10, 1, 0.0346, 0.00346) {
#my $C = 0.01492891;
#my $C = 0.0001;
#my $alpha = 500;
#my $C = 1;
#1; #0.000346;
#my $alpha = 0.01734;
#my $alpha = 1.01;
#my $C = 908;
#foreach my $T (200, 300) {
#   foreach my $alpha (1, 10, 100, 500) {
