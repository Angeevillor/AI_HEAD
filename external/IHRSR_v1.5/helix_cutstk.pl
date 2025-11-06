#!/usr/bin/perl
use warnings;
use strict;
use POSIX;

print << "EOF";

helix_cutstk v1.3 - generates stack file containing overlapping helix
segments from longer boxed helices. File names are assumed to have .dat
extension, but if extension is provided it will automatically be removed.

Input files must be in spider format and have correct byte order. This
is normally done using proc2d with the spiderswap option to convert
mrc or spi files to the proper format. For example

   proc2d infile.spi outfile.dat spiderswap

EOF


print "Enter files to be processed (wildcards allowed): ";
my $list = <STDIN>;
chomp($list);
unless($list =~ /\.dat$/) {
    $list .= ".dat";
}
my @infiles = glob("$list");

print "Enter the size of the cut boxes: ";
my $box_length = <STDIN>;
chomp($box_length);

print "Enter the shift distance: ";
my $shift = <STDIN>;
chomp($shift);

print "Enter the stacked image file to be created: ";
my $outfile = <STDIN>;
chomp($outfile);
$outfile =~ s/\.dat//;
if(-r "${outfile}.dat") {
    unlink <${outfile}.dat>;
}

open(LOG, ">helix_cutstk.log");
print LOG "input file                       start  end     count  \n";
print LOG "------------------------------   ------ ------- -------\n";

my $counter = 0;
foreach my $infile (@infiles) {

    # Skip over any unreadable files
    unless(-r $infile) {
	next;
    }

    $infile =~ s/\.dat//;
 
    my $cmdfile1 = "cmd1_${infile}";
    write_cmdfile1($cmdfile1, $infile);
    my $cmd1 = "/ultradata/shared/programs/spider/bin/spider_linux_mp_opt64 spi/dat \@${cmdfile1}";
    my $cmdout1 = `$cmd1`;
    my ($helix_width, $helix_length) = get_helix_params($cmdout1);

    my $nseg = 1 + floor(($helix_length - $box_length)/$shift);
    if($nseg < 0) {
	$nseg = 0;
    }

    print "+++++++ Processing $infile +++++++\n";
    print "helix_length = $helix_length\n";
    print "helix_width  = $helix_width\n";
    print "segments     = $nseg\n";

    my $start = $counter + 1;
    my $end = $counter + $nseg;

    printf LOG "%30s\t%7d\t%7d\t%7d\n", $infile, $start, $end, $nseg;

    for (my $i=1; $i<=$nseg; $i++) {
	$counter++;
	my $xcoor = 1;
	my $ycoor = ($i-1)*$shift + 1;
	my $cmdfile2 = "cmd2_${infile}";
	write_cmdfile2($cmdfile2, $infile, $outfile, $counter,
		       $xcoor, $ycoor, $helix_width, $box_length);
	my $cmd2 = "/ultradata/shared/programs/spider/bin/spider_linux_mp_opt64 spi/dat \@${cmdfile2}";
	my $cmdout2 = `$cmd2`;
    }
}

unlink <cmd1*>;
unlink <cmd2*>;
unlink <LOG.spi>;
close(LOG);

# ---------------------------------------------------

sub get_helix_params {
    my ($string) = @_;

    my @lines = split /\n/, $string;
    my ($width, $length);
    foreach my $line (split /\n/, $string) {
	if($line =~ /X12 SET TO/) {
	    $width = $line;
	    $width =~ s/X12 SET TO://;
	    $width =~ s/\s+//g;
	}
	if($line =~ /X13 SET TO/) {
	    $length = $line;
	    $length =~ s/X13 SET TO://;
	    $length =~ s/\s+//g;
	}
    }
    $width  = sprintf "%0.0f", $width;
    $length = sprintf "%0.0f", $length;

    return($width, $length);
}

sub write_cmdfile2 {
    my ($cmdfile2, $infile, $outfile, $counter,
	$xcoor, $ycoor, $helix_width, $box_length) = @_;
    open(CMD, ">${cmdfile2}.spi");
    print CMD "; $cmdfile2: add helix segment to image stack\n";
    print CMD "wi\n";
    print CMD "${infile}\@001\n";
    print CMD "${outfile}\@$counter\n";
    print CMD "($helix_width, $box_length)\n";
    print CMD "($xcoor, $ycoor)\n";
    print CMD "en d\n";
}

sub write_cmdfile1 {
    my ($cmdfile1, $infile) = @_;
    open(CMD, ">${cmdfile1}.spi");
    print CMD "; $cmdfile1 : Determine box dimensions\n";
    print CMD "md\n";
    print CMD "term on\n";
    print CMD "fi x12, x13\n";
    print CMD "$infile\@001\n";
    print CMD "(12,2)\n";
    print CMD "en d";
    close(CMD);
    return;
}
