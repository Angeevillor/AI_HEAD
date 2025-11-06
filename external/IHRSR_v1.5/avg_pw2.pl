#!/usr/bin/perl
use warnings;
use strict;

print "Enter power spectra to be averaged (wildcards allowed): ";
my $list = <STDIN>;
my @imgfiles = glob("$list");
foreach my $imgfile (@imgfiles) {
    $imgfile =~ s/\.dat//;
}

print "Enter name of avg power spectrum (.dat extension not needed): ";
my $avgfile = <STDIN>;
$avgfile =~ s/\s+$//;
$avgfile =~ s/\.dat//;

print "Enter box dimension (pixels): ";
my $boxdim = <STDIN>;
$boxdim =~ s/\s+$//;

# Create blank image
create_blank_image($avgfile, $boxdim);

# Accumulate images
foreach my $imgfile (@imgfiles) {
    accum_image($imgfile, $avgfile);
}

# ----------------------------------------------------

sub create_blank_image {
    my ($avgfile, $boxdim) = @_;
    
    open(CMD, ">blank.spi");
    print CMD "bl\n";
    print CMD "$avgfile\n";
    print CMD "$boxdim, $boxdim\n";
    print CMD "b\n";
    print CMD "0.0\n";
    print CMD "en d\n";
    close(CMD);
    
    my $cmd = "spider spi/dat \@blank";
    my $cmdout = `$cmd`;
}

sub accum_image {
    my ($img, $accum) = @_;
    
    open(CMD, ">accum.spi");
    print CMD "ad\n";
    print CMD "$img\n";
    print CMD "$accum\n";
    print CMD "$accum\n";
    print CMD "*\n";
    print CMD "en d\n";
    close(CMD);
    my $cmd = "spider spi/dat \@accum";
    my $cmdout = `$cmd`;
}
