FC90 = gfortran

# For gnu compilers
FFLAGS = -O2 -fconvert=swap

# For Portland group compilers
#FFLAGS = -O2 -fconvert=swap -DPGI

OBJC = spider_read.o spider_write.o cart_to_cyl.o
OBJ1 = himpose.o 
OBJ2 = hsearch_lorentz.o lsctest_lorentz.o

all: himpose hsearch_lorentz

himpose: $(OBJ1) $(OBJC)
	$(FC90) $(FFLAGS) -o himpose $(OBJ1) $(OBJC)

hsearch_lorentz: $(OBJ2) $(OBJC)
	$(FC90) $(FFLAGS) -o hsearch_lorentz $(OBJ2) $(OBJC)


clean:	
	rm  *.o himpose hsearch_lorentz

.f.o:
	$(FC90) $(FFLAGS) $(INC) -c $*.f

.F.o:
	$(FC90) $(FFLAGS) $(INC) -c $*.F

