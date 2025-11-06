#!/usr/bin/perl
use warnings;
use strict;
use POSIX;

my @infiles = @ARGV;
my $outfile = pop(@infiles);

my $count = @ARGV;
if($count < 2) {
    print "Usage\n";
    print "   merge_stack.pl instack1 [instack2] ... outstack\n";
    print "\n";
    print "File names can be entered with or without the .dat extension\n";
    print "If outstack exists, user will be prompted to either append\n";
    print "to end of file or overwrite\n";
    exit(0);
}

# String off .dat extension if provided
$outfile =~ s/\.dat//;
foreach my $infile (@infiles) {
    $infile =~ s/\.dat//;
}

if(-e "${outfile}.dat") {
    my $flag;
    do {
	print "\n";
	print "${outfile}.dat exists. Overwrite, append, or quit [o|a|q]: ";
	my $choice = <STDIN>;
	if($choice =~ /o/i) {
	    unlink "${outfile}.dat";
	    $flag = 0;
	} elsif($choice =~ /a/i) {
	    $flag = 0;
	} elsif($choice =~ /q/i) {
	    print "Exiting  merge_stack\n";
	    exit(0);
	} else {
	    print "Must enter 'o' or 'a'\n";
	    $flag = 1;
	}
    } while($flag);
}

foreach my $infile (@infiles) {
    merge_stack($infile, $outfile);
}

# ----------------------------------------

sub merge_stack {
    my ($infile, $outfile) = @_;

    # Get number of images in $infile
    open(CMD, ">temp.spd");
    print CMD "[count1]=0\n";
    print CMD "do [i]=1,10000000\n";
    print CMD "  iq fi [exist]\n";
    print CMD "  $infile\@{******[i]}\n";
    print CMD "  if([exist].gt.0) then\n";
    print CMD "    [count1]=[count1]+1\n";
    print CMD "  else\n";
    print CMD "    exit\n";
    print CMD "  endif\n";
    print CMD "enddo\n";
    print CMD "\n";

    # Get number of images in outfile
    if(-r "${outfile}.dat") {
	print CMD "[count2]=0\n";
	print CMD "do [i]=1,10000000\n";
	print CMD "  iq fi [exist]\n";
	print CMD "  $outfile\@{******[i]};\n";
	print CMD "  if([exist].gt.0) then\n";
	print CMD "    [count2]=[count2]+1\n";
	print CMD "  else\n";
	print CMD "    exit\n";
	print CMD "  endif\n";
	print CMD "enddo\n";
    } else {
	print CMD "[count2]=0\n";
    }
    print CMD "\n";
    
    print CMD "do [i]=1,[count1]\n";
    print CMD "  [j]=[count2]+[i]\n";
    print CMD "  cp\n";
    print CMD "  $infile\@{******[i]}\n";
    print CMD "  $outfile\@{******[j]}\n";
    print CMD "enddo\n";
    print CMD "\n";

    print CMD "en d\n";

    close(CMD);

    my $cmd = "spider spd/dat \@temp";
    system("$cmd");

    return;
}
