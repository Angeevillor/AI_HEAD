#!/usr/bin/perl
use warnings;
use strict;

print << "EOF";

helix_automation v1.3 - convert files to correct byte ordering, rename
files to follow spider convention, and generate 2d power spectra from
boxed helices.

Spider can only operate on files that have the correct byte ordering
and whose names contain a single '.' separating the main part of the
file name from the extension (mrc, dat, spi). The following steps are
taken to ensure that these critera are met

  o For mrc or spi files, the proc2d command is used with the spiderswap
    option to get files into the correct byte order

  o For all files, any periods in the file name before the final period
    separating the main name and extension are converted to underscores

EOF

print "Enter files to be processed (wildcards allowed): ";
my $list = <STDIN>;
my @infiles = glob("$list");

print "Enter dimension for powspec calcs (>= longest tube, default = 2048): ";
my $boxdim = <STDIN>;
if(defined $boxdim && $boxdim =~ /^\d+$/) {
    $boxdim =~ s/\s+//g;
} else {
    $boxdim = 2048;
}

print "Enter size of windowed powspec (default = previous entered value): ";
my $windim = <STDIN>;
if(defined $windim && $windim =~ /^\d+$/) {
    $windim =~ s/\s+//g;
} else {
    $windim = $boxdim;
}

if ($windim > $boxdim) {
    print "\n";
    print "Windowing dimension > power spectrum dimension\n";
    print "Ignoring input value\n";
    print "\n";
}

print "Enter pixel threshold value or 'none' for no thresholding: ";
my $threshold=<STDIN>;
unless(defined $threshold) {
    print "Need to enter a threshold value\n";
    exit(0);
}
$threshold =~ s/\s+//g;
if($threshold =~ /[a-zA-Z]/ && $threshold !~ /none/i) {
    print "Threshold is not a numerical value or 'none'\n";
    exit(0);
}


my $time_start = time();
my $file_count = 0;
my $success_count = 0;

open(LOG, ">automation.log");
foreach my $infile (@infiles) {
    if($infile eq "LOG.spi" || $infile =~ /^cmd/) {
	next;
    }

    $file_count++;
    print "\n+++++ file: $infile\n\n";
    print LOG "\n+++++ file: $infile\n\n";

    # Split filename into base and extension; convert all remaining '.' to '_'
    my $ext = $infile;
    $ext =~ s/^.*\.//;
    my $base = $infile;
    $base =~ s/\.\w{3}$//;
    $base =~ s/\./_/g;

    my $newfile = "${base}.dat";
    my $cmdfile = "cmd${base}";
    my $powfile = "pow${base}";
    my $thrfile = "thr${base}";

    if($ext ne "mrc" && $ext ne "spi" && $ext ne "dat") {
	print "\t$infile extension - must be mrc, spi, or dat\n";
	print LOG "\t$infile extension - must be mrc, spi, or dat\n";
	next;
    }

    if($ext eq "dat" && $newfile ne $infile) {
	print "\trenaming file from $infile to $newfile\n\n";
	print LOG "\trenaming file from $infile to $newfile\n\n";
	rename $infile, $newfile;
    } elsif ($ext eq "dat") {
	print "\t$infile properly named and in correct byte order\n\n";
	print LOG "\t$infile properly named and in correct byte order\n\n";
    }

    if($ext eq "mrc" || $ext eq "spi") {
	my $cmd = "proc2d $infile $newfile spiderswap";
	my $cmdout = `$cmd`;
	if($cmdout =~ /error/i) {
	    print "\t***** error processing with proc2d\n";
	    print LOG "\t***** error processing with proc2d\n";
	    next;
	} else {
	    print "\t$infile successfully converted to $newfile\n";
	    print LOG "\t$infile successfully converted to $newfile\n";
	}
    }

    unlink "${powfile}.dat";
    unlink "${thrfile}.dat";
    write_cmdfile($newfile, $cmdfile, $powfile, 
		  $thrfile, $boxdim, $windim, $threshold);	
    my $cmd = "/ultradata/shared/programs/spider/bin/spider_linux_mp_opt64 spi/dat \@${cmdfile}";
    my $cmdout = `$cmd`;

    if($threshold =~ /none/i) {
	if(-e "${powfile}.dat") {
	    print "\tpower spectrum successfully generated\n";
	    print LOG "\tpower spectrum successfully generated\n";
	} else {
	    print "\t*** error generating power spectrum\n";
	    print LOG "\t*** error generating power spectrum\n";
	    next;
	}
    } else {
	if(-e "${thrfile}.dat") {
	    print "\tpower spectrum successfully generated\n";
	    print LOG "\tpower spectrum successfully generated\n";
	} else {
	    print "\t*** error generating power spectrum\n";
	    print LOG "\t*** error generating power spectrum\n";
	    next;
	}
    }

    $success_count++;
}

my $time_end = time();
my $time_total = $time_end - $time_start;

print "\n---- calculations complete ----\n";
print "Total run time (seconds)     = $time_total\n";
print "Number of input files        = $file_count\n";
print "Number successfully processed        = $success_count\n";
print LOG "\n---- calculation complete ----\n";
print LOG "Total run time (seconds)             = $time_total\n";
print LOG "Number of input files                = $file_count\n";
print LOG "Number successfully processed        = $success_count\n";

close(LOG);

# ---------------------------------------------------

sub write_cmdfile {
    my ($newfile, $cmdfile, $powfile, 
	$thrfile, $boxdim, $windim, $threshold) = @_;
    open(CMD, ">${cmdfile}.spi");

    $newfile =~ s/\.dat//; # Make sure extension has been removed

    print CMD "; $cmdfile : PD, PW 2, AR, TH S\n";

    print CMD "PD\n";
    print CMD "$newfile\@001\n";
    print CMD "_1\n";
    print CMD "$boxdim, $boxdim\n";
    print CMD "B\n";
    print CMD "1, 1\n";

    if($windim >= $boxdim) {
	print CMD "PW 2\n";
	print CMD "_1\n";
	print CMD "$powfile\n";
    } else {
	my $corner = int(($boxdim-$windim)/2) + 1;

	print CMD "PW 2\n";
	print CMD "_1\n";
	print CMD "_2\n";
	print CMD "WI\n";
	print CMD "_2\n";
	print CMD "$powfile\n";
	print CMD "($windim, $windim)\n";
	print CMD "($corner, $corner)\n";
    }
	    

    if($threshold !~ /none/i) {
	print CMD "AR\n";
	print CMD "$powfile\n";
	print CMD "_1\n";
	print CMD "10000*p1\n";
	
	print CMD "TH S\n";
	print CMD "_1\n";
	print CMD "$thrfile\n";
	print CMD "A\n";
	print CMD "$threshold\n";
    }

    print CMD "en d\n";
    close(CMD);
    return;
}


sub leftpad {
    my ($i, $digits) = @_;
    my $len = length($i);
    my $zeros = "";
    for(my $j=1; $j<=($digits-$len); $j++) {
	$zeros .= "0";
    }
    my $padded = "${zeros}${i}";
    return $padded;
}
