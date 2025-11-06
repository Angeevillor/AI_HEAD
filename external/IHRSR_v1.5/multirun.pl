#!/usr/bin/perl
use strict;
use warnings;

my $startdir = `pwd`;
chomp($startdir);

unless(defined $ARGV[0]) {
print << "EOF";

Usage: multi_helix.pl infile

Infile must have the following format:

 dir1 script_name1
 dir2 script_name2
 ...

The directory name can be relative or absolute. The script name
can be specified with or without the .dat suffix. Any amount of 
extra whitespace is allowed and empty lines are ignored. All text
from '#' to end of line is ignored.

EOF

exit(0);
}

while (<>) {

    # Get rid of comments
    s/#.*//;

    # Skip over empty lines after comment removal
    unless(/[a-zA-Z0-9]/) {
	next;
    }

    my ($dir, $script) = split(/\s+/, $_);
    $dir =~ s/\s+//g;
    $script =~ s/\s+//g;
    $script =~ s/\.spi$//;

    unless(defined $dir && defined $script) {
	print "Directory or script not defined for input record:\n";
	print $_;
	print "\n";
	next;
    }

    unless(-e $dir) {
	print "Directory does not exist for input record:\n";
	print $_;
	print "\n";
	next;
    }

    unless(-e "$dir/$script.spi") {
	print "Script does not exist for input record:\n";
	print $_;
	print "\n";
	next;
    }

    chdir($dir);

    my $cmd = "spider spi/dat \@$script";

    print "Processing spider job:\n";
    print "dir = $dir\n"; ###
    print "cmd = $cmd\n"; ###
    print "\n";

    system($cmd);

    chdir($startdir);
}
