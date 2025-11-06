      subroutine lsctest_lorentz(x,y,a,b,c,d)
      real*8 x(21),y(21),dy(21),efx,dfx,negci(4),posci(4),fxs(21)
      real*8 prob,answers(4),variance
      real*4 returnvalue
      
      logical adflag,linear,itrctl(4)
      character*10 parmnames(4)
      character*40 filename
      
      imax=21
      
      do i=1,21
         dy(i)=1.0D0
      enddo
      
      logout=0
      adflag=.false.
      linear=.false.
      numparms=4

      do i=1,4
         itrctl(i)=.true.
      enddo
      
      numpts=imax
      citype=0
      prob=.68D0
      
      answers(1)=a
      answers(2)=b
      answers(3)=c
      answers(4)=d
      
      call lsfit2(ADFLAG, NUMPARMS, PARMNAMES, itrctl, ANSWERS, negci,
     $     posci,NUMPTS, X, Y, DY, FXS, logout, variance, ierr, CITYPE,
     $     PROB, LINEAR)

      a=answers(1)
      b=answers(2)
      c=answers(3)
      d=answers(4)

      if(ierr.gt.0)print *,ierr, (answers(i),i=1,4)

      
      end subroutine lsctest_lorentz

c**************************************************************
      real*8 function efx(ans,x,n,iflag)
      implicit none
      real*8 ans(4),x(21)
      integer n, iflag

      iflag=0
      
      efx=ans(4) + ans(1)/(1. + ( (x(n)-ans(2))/ans(3))**2)
      
      return
      end function efx

c************************************************************

      subroutine lsfit2(ADFLAG, NUMPARMS, PARMNAMES, itrctl, ANSWERS,
     $     negci, posci, NUMPTS, X, Y, DY, FXS, logout, variance, ierr,
     $     CITYPE, PROB, LINEAR)
c
c 
c THIS IS THE MAIN FITTING ROUTINE
c
c CALLING PARAMETERS
c
c      EFX, DFX, ADFLAG, NUMPARMS, PARMNAMES, ITRCTL, ANSWERS, NUMPTS, X, Y, 
C       DY, LOGOUT, CITYPE, and PROB must be specified when LSFIT is called.
c
c      ANSWERS, NEGCI, POSCI, FXS, VARIANCE and IERR are returned by LSFIT.
C
c EFX      is the (REAL*8) function to evaluate the fitting function.
c      Z = EFX(ANS,X,N,IFLAG)
c            where      
c                  ANS       (REAL*8, vector) is the current set of fitting 
c                        parameters.  See ANSWERS parameter for LSFIT
c                  X      (REAL*8, vector) is the vector of independent 
c                        variables
c                  N      (INTEGER) specifies which element of X to 
c                        evaluate the function at.
c                  IFLAG   (INTEGER) should be set to 1 if an error has
c                        occurred during the function evaluation.  This
c                        parameter is used to mark error conditions such
c                        as a division by zero or whatever.
c
c DFX      is the subroutine to evaluate the partials of the fitting function.
c      CALL DFX(ANS,ITRCTL,X,N,DERS,IFLAG)
c            where
c                  ANS, X, N, and IFLAG are the same as for EFX above
c                  ITRCTL      (LOGICAL vector) which specifies if the 
c                        corresponding ANS parameter is being optimized.
c                        .TRUE. if being optimized, .FALSE. if not.
c                  DERS      (REAL*8 vector) On return this array will contain
c                        the partials of EFX with respect to ANS(i).
c
c ADFLAG       (LOGICAL) which specifies if analitical partials, i.e. the
c            DFX routine is to be used.  If this is .FALSE. then the routine
c            will ignore the DFX routine and evaluate the partials 
c            numerically.  I.E., if .FALSE. then DFX can be a dummy routine.
c
c NUMPARMS       (INTEGER) which specifies the number of parms being estimated.
c            Must be greater than zero.
c
c PARMNAMES      (CHARACTER*10 vector) is the corresponding names of the 
c            parameters being estimated.  If LOGOUT is set to zero then these
c            names are not used by the routine.
c
c ITRCTL      (LOGICAL vector) is a vector which specifies which of the 
c            parameters are to be estimated.  There is a one to one 
c            correspondence between this vector and the PARMNAMES, ANSWERS,
c            NEGCI and POSCI vectors.
c
c ANSWERS      (REAL*8 vector) is a vector containing: Initially will contain
c            the "guesses" of the parameters to be estimated and a constant
c            value for any parameters not currently being estimated.  On 
c            return this vector will contain the parameter values with the
c            highest probability of being correct.
c
c NEGCI            (REAL*8 vector) on return will contain the lower limit of the
c            confidence interval of the estimated parameters.  See also
c            CITYPE and PROB below.
c
c POSCI            (REAL*8 vector) on return will contain the upper limit of the
c            confidence interval of the estimated parameters.  See also
c            CITYPE and PROB below.
c
c NUMPTS      (INTEGER) contains the number of data points.
c
C X            (REAL*8 vector) contains the independent variables of the data.
c
C Y            (REAL*8 vector) contains the dependent variables of the data.
c
C DY            (REAL*8 vector) contains the standard deviations of the 
C            independent variables.  Must all be greater than zero.  For an
c            unweighted fit set DY(i)=1.0D0, for all i.
c
C FXS            (REAL*8 vector) will contain, on return, the weighted residuals
c            of the fit.  I.e., (Y(i)-EFX(i))/DY(i) for all i, where EFX(i) is
c            the fitted function evaluated at the i-th data point and the 
c            ANSWERS with the highest probability of being correct.
c
C LOGOUT      (INTEGER) specifies the FORTRAN output unit to write the summary
c            file to.  If LOGOUT=0 no file will be written.
c
C VARIANCE      (REAL*8) on return will contain the weighted variance of the 
c            fit.
c
C IERR            (INTEGER) is the return error flag.
c            If IERR < 0 then estimated parameters are O.K., but confidence
c                  intervals are not O.K.
c            If IERR = 0 then everything is O.K.
c            If IERR > 0 then estimated parameters and confidence intervals
c                  are wrong
c
c            If IERR=-7 The user aborted the fit by hitting the ESCAPE
c                  key during the confidence interval search.  Parmater
c                  values are O.K.
c             If IERR=-6 Number of degrees of freedom (NDF) was equal to 
c                  zero.  The program cannot search for a significant
c                  increase in the variance under with no degrees of
c                  freedom.  Try CITYPE=1 or 2. Parameter values are
c                  O.K.
c             If IERR=-5 an error occurred while searching a parameter for 
c                  a significant increase in the Variance to define a
c                  CITYPE=3 confidence interval.  Parameter values are
c                  O.K.  Try CITYPE=1 or 2. This error probably means that
c                  the variance space was flat, i.e. a highly correlated
c                  system.
c             If IERR=-4 an error occurred while searching an eigenvector for 
c                  a significant increase in the Variance to define a
c                  CITYPE=3 confidence interval.  Parameter values are
c                  O.K.  Try CITYPE=1 or 2. This error probably means that
c                  the variance space was flat, i.e. a highly correlated
c                  system.
c             If IERR=-3 one of the eigenvalues was not greater than 0.0D0. 
c                  Parm values are o.k.  Try CITYPE=1 or 2.
c            If IERR=-2 could not evaluate F-Statistic, Parm Values are O.K.
c                  PROB is probably wrong.
c             If IERR=-1 could not evaluate the Eigenvalues and Eigenvectors
c                  of the Correlation Matrix, Parm values o.k.  Try 
c                  CITYPE=1 or 2.
c            If IERR= 1 too many parameters!  Must recompile with larger 
c                  arrays.
c            If IERR= 2 return because no parms to be fit. ITRCTL all .FALSE.
C            If IERR= 3 The function could not be evaluated at the initial
C                  parameter values in the ANSWERS array.
c             If IERR= 4 illegal weighting factors.  An element of the DY 
c                  array was less than or equal to zero.
c             If IERR= 5 too many Gauss-Newton loops without convergences.
C            IF IERR= 6 an error occurred in the evaluation of the partials.
c            If IERR= 7 error in SYMSV matrix solution routine.  This 
c                  probably means that the parameters are infinitely
c                  correlated.  I.e. you are attempting to get too much
c                  information from the data.
c             If IERR= 8 too many inner loops without convergence.  I do not
c                  think that this error can ever happen, but I included a
c                  counter in the loop so that it could not become an 
c                  infinite loop.
c            If IERR= 9 The user aborted the program by hitting an ESCAPE
c                  key during the initial convergence.  The parameter 
c                  values are wrong.
c
c CITYPE specifies which type of confidence intervals are desired.  Usually,
c      this should be set to 3 for a searched nonlinear joint confidence
c      interval with a probability given by PROB.  
c      CITYPE=0 will evaluate no confidence intervals.
c      CITYPE=1 will provide intervals corresponding to the linear 
c            asymptotic standard errros of the parameters.  
c      CITYPE=2 will multiply CITYPE by the largest eigenvalue of the 
c            correlation matrix to provide intervals which include an 
c            approximation of the covariance between the parameters.
c      CITYPE=3, Usual value.
c
c PROB      specifies the probability level for CITYPE=3 confidence intervals.
c      For one standard deviation PROB=0.68, for two standard deviations
c      PROB=0.95, etc.
c
c LINEAR (logical) specifies if a linear least-squares fit is being performed.
c
c declarations of the calling parms 
      integer CITYPE
c if =0, do no confidence intervals
c if =1, do asymptotic confidence intervals
c if =2, do linear joint confidence intervals
c if =3, search space for nonlinear joint confidence intervals
c
      external EFX,DFX
      real*8 EFX
      logical ADFLAG
      integer numparms,numpts,logout
      character*10 parmnames(numparms)
      logical itrctl(numparms),linear
      real*8 answers(numparms),negci(numparms),posci(numparms)
      real*8 x(numpts),y(numpts),dy(numpts)
      real*8 variance
      integer ierr
c end of calling sequence declarations
c
c two location to store the sum of squares for each iterations
      real*8 oldssr,newssr,bstssr,ZZ,FZQ1,FZQ2,sumsq
      real*8 FSTMJ
      integer ndf,convloop
      real*8 prob
c
c storage for the current function values
      real*8 fxs(numpts)
c
c storage for the current partials of function with respect to the
c fitting parameters
      real*8 partials
      allocatable partials(:)
c
c current number of loops 1 and 2
      integer cur1loop,cur2loop
c
c pointer to which parms are being fit and number
      integer pointer,numfit
      allocatable pointer(:)
c
c storage for the matrix equation ATA*E=ATD
      real*8 ATA,ATD,E
      allocatable ATA(:,:),ATD(:),E(:)
c
c tempary answers
      real*8 bstans,tmpans,temp
      allocatable bstans(:),tmpans(:)
c
c scratch arrays for matrix inversion
      real*8 T,S,scratch
      allocatable T(:),S(:,:),scratch(:,:)
c
      integer i,j,IFLAG,n,ii,ipt,IERRORS
      real*8 z,weight,dif
c
      allocate(partials(numparms), bstans(numparms), tmpans(numparms))
c
c initialize error return
      ierr=0

c
      max_loops=100
      converge_loops=1
c


c
c if =-6 cannot find searched confidence intervals if number of degrees of
c if =-5 error searching parameter for C.I., Parms are O.K.
c if =-4 error searching eigenvector for C.I., Parms are O.K.
c if =-3 an eigenvalues was not greater than 0, Parm values are o.k.
C if =-2 could not evaluate F-Statistic, Parm Values are O.K.
c if =-1 could not evaluate Eigenvalues of Correlation Matrix, Parm values o.k.
c if = 0 normal return
c if = 1 more parameters than array sizes allow
c if = 2 return because no parms to be fit
c if = 3 could not evaluate function at the initial values
c if = 4 illegal weighting factors
c if = 5 too many Gauss-Newton loops without convergences
c if = 6 error in evaluating partials
c if = 7 error in symsv routine
c if = 8 too many inner loops without convergence
c        freedom equals zero
c
c set up pointers to fitted parms
      numfit=0
      do i=1,numparms
         if(.not.itrctl(i)) cycle
         numfit=numfit+1
      enddo

      allocate (T(numfit),S(numfit,numfit),scratch(numfit,numfit))
      allocate (ATA(numfit,numfit),ATD(numfit),E(numfit))
      allocate (pointer(numfit))
      numfit=0
      do i=1,numparms
         if(.not.itrctl(i)) cycle
         numfit=numfit+1
         pointer(numfit)=i
      enddo

c     write(logout,*) ' pointer',(pointer(i),i=1,numfit)
c
c     check to see if any fit is actually being performed
      if(numfit.le.0) then
         ierr=2
         deallocate (pointer)
         deallocate (ATA,ATD,E)
         deallocate (T,S,scratch)
         deallocate (partials,bstans,tmpans)
         return
      endif
c
c evaluate oldssr and fxs (the current function values)
      oldssr=0.0D0
      IFLAG=0
      do i=1,numpts
         n=i
         z=EFX(answers,x,N,IFLAG)
         if(IFLAG.eq.1) then
            ierr=3
            deallocate (pointer)
            deallocate (ATA,ATD,E)
            deallocate (T,S,scratch)
            deallocate (partials,bstans,tmpans)
            return
         endif
         fxs(i)=z
c     
c     check for valid weights, i.e. must be greater than zero
         if(dy(i).le.0.0D0) then
            ierr=4
            deallocate (pointer)
            deallocate (ATA,ATD,E)
            deallocate (T,S,scratch)
            deallocate (partials,bstans,tmpans)
            return
         endif
         z=(fxs(i)-y(i))/dy(i)
         oldssr=oldssr+z*z
      enddo
c     write(logout,*) ' FXS', (FXS(i),i=1,numpts)
c     write(logout,*) ' OLDSSR',oldssr
c
c
c store current best set of answers
      bstssr=oldssr
      do i=1,numparms
         bstans(i)=answers(i)
      enddo
      cur1loop=0
c     initialize counter for number of major loops
c     
      convloop=0
c     initialize counter for number of times convergence test has passed
c     
c     loop back to here for main convergence loop
 1000 continue
c     
c     test for toooo many iterations
      call working(*97345)
      cur1loop=cur1loop+1
      if(cur1loop.gt.max_loops) then
         ierr=5
         go to 3900
      endif
c     
c     write(logout,1001) cur1loop,oldssr
c     1001      format(1x,i5,1pD15.5)
c     
c     store answers in temparary array
      do i=1,numparms
         tmpans(i)=answers(i)
      enddo
c     if(logout.gt.0) then
c     write(logout,1011) oldssr,(answers(i),i=1,numparms)
c     endif
c1011      format(1x,1pD15.5,4x,3D15.5,:,/,(20x,3D15.5,:))
c
c now initialize the matrix equation to all zeros
      do i=1,numfit
         do j=1,numfit
            ATA(i,j)=0.0D0
         enddo
         ATD(i)=0.0D0
      enddo
c
c now calculate ATA and ATD
      IFLAG=0
      do ipt=1,numpts
c     must loop over each data point
c     
         weight=dy(ipt)
c     
c     calculate the weighted difference between data and calculated curve
         dif=(y(ipt)-fxs(ipt))/weight
c     
CCCCC if (ADFLAG) then
c     
c     here if user supplied parital derivatives
CCCCCCcall dfx(answers,itrctl,X,ipt,partials,IFLAG)
CCCCCCelse
c     
c     here if numerical parital derivatives
         call dfxnum(answers,numparms,itrctl,X,ipt,
     *        partials,EFX,FXS,IFLAG,numpts)
CCCC  endif
c     
c     check for errors in the evaluation of partials
         if(IFLAG.ne.0) then
            ierr=6
            go to 3900
         endif
c     
c     devide partiasl by weighting factors
         do i=1,numfit
            ii=pointer(i)
            partials(ii)=partials(ii)/weight
         enddo
c     
c add terms for a single data point to matrix equation
         do i=1,numfit
            ATD(i)=ATD(i)+dif*partials(pointer(i))
            do j=i,numfit
               ATA(i,j)=ATA(i,j)+
     *              partials(pointer(i))*partials(pointer(j))
            enddo
         enddo
c     
c fill in remainder of ATA matrix
         do i=1,numfit
            do j=i,numfit
               ATA(j,i)=ata(i,j)
            enddo
         enddo
c     
      enddo
c
c now solve matrix equation by sqrt root method
      IFLAG=1
      call symsv(ATA,E,ATD,numfit,IFLAG,S,T)
c     
c     return if error in matrix solver
      if(IFLAG.eq.0) then
         ierr=7
         go to 3900
      endif
c     
c update answer array
      do i=1,numfit
         ii=pointer(i)
         answers(ii)=answers(ii)+E(i)
      enddo
c
      cur2loop=0
c initialize diverence correction loop counter
c
c loop back to here if step was too big
 3000 continue
c
c check for too main inner loops
      cur2loop=cur2loop+1
      call working(*97345)
      if(cur2loop.gt.max_loops) then
         ierr=8
         deallocate (pointer)
         deallocate (ATA,ATD,E)
         deallocate (T,S,scratch)
         deallocate (partials,bstans,tmpans)
         call e_working
         return
      endif
c
c calculate new SSR
      newssr=0.0D0
      IFLAG=0
      do i=1,numpts
         n=i
         z=EFX(answers,x,N,IFLAG)
c     
c     if error then answers must be too far off, 
c     branch to devide correction vector
         if(IFLAG.eq.1) then
            newssr=10.0*oldssr
            go to 3500
         endif
         fxs(i)=z
         z=(fxs(i)-y(i))/dy(i)
         newssr=newssr+z*z
      enddo
c     
c     if better answers then store them
      if(newssr.lt.bstssr) then
         bstssr=newssr
         do i=1,numparms
            bstans(i)=answers(i)
         enddo
      endif
c     
c     write(logout,3001) cur2loop,newssr
c     3001      format(21x,i5,1pD15.5)
c     
c     now check for convergence
      if(linear) go to 4000
c     
      Zz=abs(newssr/oldssr-1.0D0)
c     write(logout,3002) ZZ
c     3002      format(41x,1pD15.5)
      if(ZZ.GT.0.00001D0) GO TO 3500
c     branch because newssr changed too much
c     
      do i=1,numfit
         ii=pointer(i)
         if(tmpans(ii).eq.0.0D0) cycle
         Zz=dabs(answers(ii)/tmpans(ii)-1.0D0)
c     write(logout,3003) ii,ZZ
c     3003      format(56x,i5,1pD15.5)
         if(Zz.gt.0.00001D0) go to 3500
c     branch because parameters changed too much
c     
      enddo
c     
c     check for correct number of passed convergenge tests in a row
      convloop=convloop+1
c     write(6,*) ' convloop',convloop
      if(convloop.lt.converge_loops) then
         go to 1000
      else
         go to 4000
      endif
c
c     here to decrease correction vector
 3500 continue
c     
c     if SSR decreased then go do it all over again
      if(newssr.lt.oldssr) then
         oldssr=newssr
         convloop=0
         go to 1000
      endif
c     if(logout.gt.0) then
c     write(logout,1012) newssr,(answers(i),i=1,numparms)
c     endif
c     1012      format(1x,4x,1pD15.5,3D15.5,:,/,(20x,3D15.5,:))
c     
c     shrink the vector
      do i=1,numfit
         ii=pointer(i)
         E(i)=E(i)*0.5D0
         answers(ii)=answers(ii)-E(i)
      enddo
c     
c go back to inner loop
      go to 3000
c     
c     here if some error occurred, reset the answer array
 3900 continue
      do i=1,numparms
         answers(i)=bstans(i)
      enddo
      deallocate (pointer)
      deallocate (ATA,ATD,E)
      deallocate (T,S,scratch)
      deallocate (partials,bstans,tmpans)
      call e_working
      return
c
c
 4000 continue
c     here for convergence
c     
      if(bstssr.lt.newssr) then 
c     
c     reset to very best answers found so far
c     
         newssr=bstssr
         do i=1,numparms
            answers(i)=bstans(i)
         enddo
c     
c     now recalculate the function at those answers
         do i=1,numpts
            n=i
            z=EFX(answers,x,N,IFLAG)
c     no need to test for errors since it passed the test once
            fxs(i)=z
         enddo
c     
      endif
c
c     calculate the variance
      if(numfit.eq.numpts) then
         variance=0.0D0
      else
         variance=newssr/(numpts-numfit)
      endif
c     
c     calculate the number of degrees of freedom
      ndf=numpts-numfit
c     
      if(logout.gt.0) then
c     write(logout,1011) newssr,(answers(i),i=1,numparms)
         write(logout,4013) variance,NDF
 4013    format(//,' Variance of Fit =',1pg15.5,///
     *        ' Number of Degrees of Freedom =',i6)
C     #ifdef debug
c     write(logout,4011)
c     4011            format(//,' Information Matrix')
c     do 4010 i=1,numfit
c     ii=pointer(i)
c     write(logout,4012) Parmnames(ii),(ATA(i,j),j=1,i)
c     4010            continue
C     #endif
      endif
c     
c     initialize the confidence interval arrays
      do i=1,numfit
         ii=pointer(i)
         negci(ii)=answers(ii)
         posci(ii)=answers(ii)
      enddo
c     
c calculate the residuals
      do i=1,numpts
         fxs(i)=(y(i)-fxs(i))/dy(i)
      enddo
c     
c     stop here if no error statistics are required
      if(CITYPE.eq.0) then
         deallocate (pointer)
         deallocate (ATA,ATD,E)
         deallocate (T,S,scratch)
         deallocate (partials,bstans,tmpans)
         call e_working
         return
      endif
c     
c now get inverse of info matrix and variance-covariance matrix
c one column at a time
c     call symsv(ATA,E,ATD,numfit,IFLAG,S,T)
      do i=1,numfit
         do j=1,numfit
            ATD(j)=0.0D0
         enddo
         ATD(i)=1.0D0
         call symsv(ATA,E,ATD,numfit,IFLAG,S,T)
         do j=1,numfit
            scratch(j,i)=E(j)
         enddo
      enddo
c
c#ifdef debug
c      if(logout.gt.0) then
c            write(logout,4201) 
c4201            format(//' Inverse of Information Matrix')
c            do 4202 i=1,numfit
c            ii=pointer(i)
c            write(logout,4012) Parmnames(ii),(SCRATCH(i,j),j=1,i)
c4202            continue
c            endif
c#endif
c
      do i=1,numfit
         do j=1,numfit
            scratch(i,j)=scratch(i,j)*variance
         enddo
      enddo
c     
c now calculate correlation matrix
      do i=1,numfit
         do j=1,numfit
            S(i,j)=scratch(i,j)/sqrt(scratch(i,i))/sqrt(scratch(j,j))
         enddo
      enddo
c
      if(logout.gt.0) then
c     #ifdef debug
c     write(logout,4211) 
c     4211            format(//' Variance-Covariance Matrix')
c     do 4212 i=1,numfit
c     ii=pointer(i)
c     write(logout,4012) Parmnames(ii),(SCRATCH(i,j),j=1,i)
c     4212            continue
c     #endif
         write(logout,4213) 
 4213    format(//' Correlation Matrix')
         do i=1,numfit
            ii=pointer(i)
            write(logout,4012) Parmnames(ii),(S(i,j),j=1,i)
 4012       format(1x,a10,2x,1p4g15.5,:,/,(13x,4g15.5,:))
         enddo
      endif
c     
c stop here for type 1 error stats      
      if(CItype.eq.1) then
         do i=1,numfit
            ii=pointer(i)
            temp=sqrt(scratch(i,i))
            negci(ii)=answers(ii)-temp
            posci(ii)=answers(ii)+temp
         enddo
         deallocate (pointer)
         deallocate (ATA,ATD,E)
         deallocate (T,S,scratch)
         deallocate (partials,bstans,tmpans)
         call e_working
         return
      endif
c
c now find eigenvalues of correlation matrix
c
      call tred2(numfit,numfit,ATD,T,S)
      call tql2(numfit,numfit,ATD,T,S,ierrors)
c     
      if((ierrors.ne.0).and.(logout.gt.0)) then
         write(logout,4301) ierrors
 4301    format(//,' TQL2 error code ',i5)
      endif
c
      if(ierrors.ne.0) then
         ierr=-1
         deallocate (pointer)
         deallocate (ATA,ATD,E)
         deallocate (T,S,scratch)
         deallocate (partials,bstans,tmpans)
         call e_working
         return
      endif
c     
      if(logout.gt.0) then
         write(logout,4302) (ATD(i),i=1,numfit)
 4302    format(//,' Eigenvalues of the Correlation Matrix',/,
     *        1x,1p5g15.5,:,/,(16x,4g15.5,:))
c     #ifdef debug
c     write(logout,4358)
c     4358            format(//,' Eigenvectors of the Correlation Matrix')
c     do 4357 i=1,numfit
c     write(logout,4316) (S(j,i),j=1,numfit)
c     4316                  format(/,1x,1p5g15.5,:,/,(16x,4g15.5,:))
c     4357            continue
c     #endif
      endif
c
c stop here for type 2 error stats
      if(CITYPE.eq.2) then
         do i=1,numfit
            ii=pointer(i)
            TEMP=SQRT(SCRATCH(I,I))*atd(NUMFIT)
c     note ATD(NUMFIT) contains larges eigenvalue of correlation matrix
            NEGCI(II)=ANSWERs(II)-TEMP
            POSCI(II)=ANSWERs(II)+TEMP
         enddo
         call e_working
         RETURN
      ENDIF
c     
c we are now ready to calculate confidence intervals
c
c
c flag an error if no degrees of freedom
      if(ndf.eq.0) then
         ierr=-6
         deallocate (pointer)
         deallocate (ATA,ATD,E)
         deallocate (T,S,scratch)
         deallocate (partials,bstans,tmpans)
         call e_working
         return
      endif
c     
c
c first find desired critical increase level for confidence intervals
c
      FZQ1=1.0D0+FLOAT(numfit)/float(ndf)*
     *     fstmj(numfit,ndf,prob,IERRORS,logout)
c get F for an added parameter
c
      IF(ierrors.NE.0) THEN
         ierr=-2
         deallocate (pointer)
         deallocate (ATA,ATD,E)
         deallocate (T,S,scratch)
         deallocate (partials,bstans,tmpans)
         call e_working
         return
      endif
      IF(logout.gt.0) then
         write(logout,4101) numfit,ndf,
     *        numfit,ndf,prob,FZQ1
 4101    format(/,' 1.0 + (',i5,'/',i5,')*F(',i5,',',i5,
     *        ',   ',f4.2,')=',1pG15.5)
      endif
c     
c     get other F
      FZQ2=fstmj(ndf,ndf,prob,ierrors,logout)
      IF(ierrors.NE.0) THEN
         ierr=-1
         deallocate (pointer)
         deallocate (ATA,ATD,E)
         deallocate (T,S,scratch)
         deallocate (partials,bstans,tmpans)
         call e_working
         return
      endif
      IF(LOGOUT.gt.0) then
         write(logout,4102) ndf,ndf,prob,FZQ2
 4102    format(/,' F(',i5,',',i5,
     *        ',   ',f4.2,')=',1pG15.5)
      endif
c     
c     take smallest F
      if(FZq1.gt.FZQ2) FZQ1=fzq2
      if(logout.gt.0) then
         write(logout,4103) prob,fzq1
 4103    format(/,' A probability of ',F4.2,' corresponds to a',
     *        ' fractional change in variance',/,' of ',1pG15.5)
      endif
c     
c     multiple Eigenvectors by square root of eigenvalues
      do i=1,numfit
         if((ATD(i).le.0.0D0).and.(LOGOUT.GT.0)) then
            write(logout,4397) numfit
 4397       format(//,' One, or more, of the ',
     *           'Eigenvalues was Zero: ',i5)
         endif
         if(ATD(i).le.0.0D0) then
            ierr=-3
            deallocate (pointer)
            deallocate (ATA,ATD,E)
            deallocate (T,S,scratch)
            deallocate (partials,bstans,tmpans)
            call e_working
            return
         endif
         temp=sqrt(ATD(i))
         do j=1,numfit
            s(j,i)=s(j,i)*temp
         enddo
      enddo
c     
c
      ierrors=0
      do i=numfit,1,-1
c     loop over each of the eigenvectors and search both directions for F
c     
c     set up tmpans as direction vector for search
         do j=1,numparms
            tmpans(j)=0.0D0
         enddo
         sumsq=0.0D0
         do j=1,numfit
c     
c     move particular eigenvector into Tmpans array
            ii=pointer(j)
            z=S(j,i)*sqrt(scratch(j,j))
            tmpans(ii)=z
            sumsq=sumsq+z*z
         enddo
         if(sumsq.eq.0.0D0) then
            ierrors=1
         else
            call searche(answers,tmpans,negci,
     *           posci,numparms,
     *           x,y,dy,numpts,EFX,ierrors,
     *           logout,fzq1,bstssr,*97346)
         endif
c     
         if((ierrors.ne.0).and.(logout.gt.0)) then
            write(logout,5002) i
 5002       format(' SEARCHE error for eigenvector',i6)
         endif
         if(ierrors.ne.0) then
            ierr=-4
            deallocate (pointer)
            deallocate (ATA,ATD,E)
            deallocate (T,S,scratch)
            deallocate (partials,bstans,tmpans)
            call e_working
            return
         endif
c     
c     change sign of search vector  and do it again
         do j=1,numparms
            tmpans(j)=-tmpans(j)
         enddo
         call searche(answers,tmpans,negci,posci,numparms,
     *        x,y,dy,numpts,EFX,ierrors,logout,fzq1,bstssr,*97346)
         if((ierrors.ne.0).and.(logout.gt.0)) then
            write(logout,5002) i
         endif
         if(ierrors.ne.0) then
            ierr=-4
            deallocate (pointer)
            deallocate (ATA,ATD,E)
            deallocate (T,S,scratch)
            deallocate (partials,bstans,tmpans)
            call e_working
            return
         endif
      enddo
c
      if(numfit.eq.1) then
         deallocate (pointer)
         deallocate (ATA,ATD,E)
         deallocate (T,S,scratch)
         deallocate (partials,bstans,tmpans)
         call e_working
         return
      endif
c     skip out if only a one parm fit, no need to search it twice
c     
c now search parms themselves
c
c start search at current estimate of confidence interval
      do ii=1,numfit
         i=pointer(ii)
         do j=1,numparms
            tmpans(j)=0.0D0
         enddo
         tmpans(i)=posci(i)-answers(i)
c     
c     make sure tmpans(i) is not equal to zero
         if(tmpans(i).eq.0.0D0) tmpans(i)=answers(i)-negci(i)
         if(tmpans(i).eq.0.0D0) tmpans(i)=0.0001D0*answers(i)
         if(tmpans(i).eq.0.0D0) tmpans(i)=0.0001D0
c     
c     now do search
         call searche(answers,tmpans,negci,posci,numparms,
     *        x,y,dy,numpts,EFX,ierrors,logout,fzq1,bstssr,*97346)
         if((ierrors.ne.0).and.(logout.gt.0)) then
            write(logout,6002) i
 6002       format(' SEARCHE error for parameter',i6)
         endif
         if(ierrors.ne.0) then
            ierr=-5
            deallocate (pointer)
            deallocate (ATA,ATD,E)
            deallocate (T,S,scratch)
            deallocate (partials,bstans,tmpans)
            call e_working
            return
         endif
         tmpans(i)=-tmpans(i)
c     
c     now do search
         call searche(answers,tmpans,negci,posci,numparms,
     *        x,y,dy,numpts,EFX,ierrors,logout,fzq1,bstssr,*97346)
         if((ierrors.ne.0).and.(logout.gt.0)) then
            write(logout,6002) i
         endif
         if(ierrors.ne.0) then
            ierr=-5
            deallocate (pointer)
            deallocate (ATA,ATD,E)
            deallocate (T,S,scratch)
            deallocate (partials,bstans,tmpans)
            call e_working
            return
         endif
      enddo
c
c     all done
      deallocate (pointer)
      deallocate (ATA,ATD,E)
      deallocate (T,S,scratch)
      deallocate (partials,bstans,tmpans)
      call e_working
      return
97345 continue
      ierr=9
      deallocate (pointer)
      deallocate (ATA,ATD,E)
      deallocate (T,S,scratch)
      deallocate (partials,bstans,tmpans)
      call e_working
      return
97346 continue
      ierr=-7
      deallocate (pointer)
      deallocate (ATA,ATD,E)
      deallocate (T,S,scratch)
      deallocate (partials,bstans,tmpans)
      call e_working
      return
      end subroutine lsfit2


      subroutine searche(answers, tmpans, negci, posci, numparms, x, y,
     $     dy, numpts, EFX, ierrors, logout, DVAR, bstssr, *)
c
C THIS ROUTINE IS USED TO SEARCH PARAMETERS AND EIGENVECTORS FOR CONFIDENCE
C INTERVALS
c
      integer numparms,numpts
      real*8 answers(numparms),tmpans(numparms),
     *     negci(numparms),posci(numparms)
      real*8 x(numpts),y(numpts),dy(numpts)
      external EFX
      real*8 EFX
      integer ierrors,logout
      real*8 DVAR,bstssr
c     
c     
      integer i,nloop,ierror
      real*8 dist,dlow,dhigh,temp,derr,dv,Z
      allocatable temp(:)
      real*8 SSR
      allocate (temp(numparms))
c     
c     nloop is the current number of loops
      nloop=0
c     
c     dlow is the lower acceptable distance
      dlow=0.0D0
c     
c     calculate the uncertainty range for acceptable DV
      derr=(DVAR-1.0D0)*0.003*bstssr
c     
c     DV is the critical level in terms of SSR
      dv=DVAR*BSTSSR
c     write(logout,*) dv,derr
c     
      DIST=1.0D0
c     initialize dist
c     
 1000 continue
      call working(*99999)
c     
c     heck for tooooooo many loops
      nloop=nloop+1
      if(nloop.gt.max_loops) then
         ierrors=1
         deallocate (temp)
         return
      endif
c     
c     create a TEMP answer array
      do i=1,numparms
         temp(i)=answers(i)+dist*tmpans(i)
c     
c     loop back to here if too large a distance has not been found
      enddo
c     write(logout,887) dist
c887      format(1x,1pg15.5)
      ierror=0
      Z=ssr(temp,x,y,dy,numpts,EFX,ierror)
c     find SSR
c
c      if(logout.gt.0) write(logout,888) Z,(temp(i),i=1,numparms)
      if(IERROR.NE.0) GO TO 2000
c if error in evaluateing SSR assume that it is too large
c
      if(dabs(Z-dv).lt.derr) go to 3000
c     check to see if we are close enough
c     
      if(Z.gt.DV) go to 2000
c     if Z is too big jump to area for too big a vaiance
c
c
c     here if dist is too small
      dlow=dist*DIST
      dist=dist*2.0D0
c     
c     now update C.I.s
      do i=1,numparms
         if(temp(i).lt.negci(i)) negci(i)=temp(i)
         if(temp(i).gt.posci(i)) posci(i)=temp(i)
      enddo
      go to 1000
c
c here when too big a distance is first found
 2000 CONTINUE
      dhigh=dist*dist
c     
c     loop back to here for interval division routine
 2100 continue
c     
c     check for too many loops
      nloop=nloop+1
      if(nloop.gt.max_loops) then
         ierrors=1
         deallocate (temp)
         return
      endif
c     
c     might never branch here but ...
      if(dlow.gt.dhigh) then
         ierrors=1
         deallocate (temp)
         return
      endif
c     
c     if change in dist goes to zero, assume that it is a valid convergence
      if(DABS( (dhigh-dlow)/(dlow+dhigh) ).lt.1.0D-5) go to 3000
c     
c     get new distance
      dist=sqrt((dhigh+dlow)/2.0D0)
c     write(logout,887) dist
c     
c     create TEMP answer array
      do i=1,numparms
         temp(i)=answers(i)+dist*tmpans(i)
      enddo
c
c calculate SSR
      ierror=0
      Z=ssr(temp,x,y,dy,numpts,EFX,ierror)
c     if(logout.gt.0) write(logout,888) Z,(temp(i),i=1,numparms)
c     
c branch if close enough
      if(dabs(Z-dv).lt.derr) go to 3000
c     
      if(Z.gt.dv) then
c     here if Z too big
c     
         dhigh=dist*dist
         go to 2100
      endif
      if(IERROR.NE.0) GO TO 2000
c     if error Z must be too big
c     
      dlow=dist*dist
c     
c     update C.I.s
      do i=1,numparms
         if(temp(i).lt.negci(i)) negci(i)=temp(i)
         if(temp(i).gt.posci(i)) posci(i)=temp(i)
      enddo
c
c go done it again
      go to 2100
 3000 continue
 888  format(16x,1p4g15.5)
c      if(logout.gt.0) write(logout,888) Z,(temp(i),i=1,numparms)
c
c update C.I.s
      do i=1,numparms
         if(temp(i).lt.negci(i)) negci(i)=temp(i)
         if(temp(i).gt.posci(i)) posci(i)=temp(i)
      enddo
      deallocate (temp)
      return
99999 continue
      return 1
      end subroutine searche


      real*8 function SSR(answers,x,y,dy,numpts,EFX,ierror)
c     
C     THIS ROUTINE CALCULATES THE SSR
c     
      integer numpts
      real*8 answers(*)
      real*8 x(numpts),y(numpts),dy(numpts)
      external EFX
      real*8 EFX
      integer ierror,ierr
c     
      real*8 sum,temp
      integer i
c     
      sum=0.0D0
      ierr=0
      do i=1,numpts
         temp=(Y(i)-EFX(answers,x,i,ierr))/dy(i)
         if(ierr.ne.0) then
            ierror=1
            ssr=1.0D300
            return
         endif
         sum=sum+temp*temp
      enddo
      ssr=sum
      return
      end function SSR


      SUBROUTINE SYMSV(A,V,F,N,IFLAG,S,T)
C
C
C SOLVES THE SYMMETRIC
C SOLVES THE MATRIX EQUATION A*V=F WHERE A IS SYMMETRIC
C VIA THE SQUARE ROOT METHOD
C
C
      integer N,i,j,im1,L,K,IFLAG,iii
      REAL*8 SUM,SUM2
      real*8 V(N)
      real*8 A(N,N),T(N),
     *     F(N),S(N,N)
      IF (A(1,1) .EQ. 0.0D0) GO TO 100
      IF(N.EQ.1) GO TO 1000
      S(1,1) = SQRT(DABS(A(1,1)))
      DO J=2,N
         S(1,J) = A(1,J)/S(1,1)
      enddo
C     
      DO I=2,N
         IM1 = I-1
         DO J=I,N
            sum2=0.0D0
            do iii=1,im1
               sum2=sum2+s(iii,i)*s(iii,J)
            enddo
            SUM=A(I,J)-SUM2
            IF (J.NE.I) GO TO 30
            if(sum < 0.0) then
               sum = -sum
               s(i,i) = sqrt(sum)
            else if (sum == 0.0) then
               go to 100
            else 
               s(i,i) = sqrt(sum)
            endif
            cycle
 30         S(I,J) = SUM/S(I,I)
         enddo
      enddo
C
 50   T(1) = F(1)/S(1,1)
      DO I=2,N
         sum2=0.0D0
         do iii=1,(i-1)
            sum2=sum2+S(iii,I)*T(iii)
         enddo
         T(I)=(F(I)-SUM2)/S(I,I)
      enddo
C
      J = N
      V(J) = T(J)/S(J,J)
 70   L=J
      J = J - 1
      IF (J.EQ.0) RETURN
C...  THE FOLLOWING SUMMATION SHOULD BE PERFORMED IN DOUBLE PRECISION
      SUM=0.0D0
      DO K=L,N
         SUM=SUM+S(J,K)*V(K)
      enddo
      V(J)=(T(J)-SUM)/S(J,J)
      GO TO 70
C     
 100  CONTINUE
      IFLAG = 0
      RETURN
 1000 CONTINUE
      V(1)=F(1)/A(1,1)
      RETURN
      END SUBROUTINE SYMSV


      real*8 FUNCTION FSTMJ(N,M,P,ierr,logout)
C     
C     THIS ROUTINE EFALUATES AN F-STATISTIC (N,M,p)
C     
      real*8 v1,v2,xp,t,a,b,h,lambda,w,z1,z2
      real*8 x,f,fl,fh,fstatz
      REAL*8 P
      INTEGER IERR,logout,m,n
      
C     
C     FIRST STEP IS TO EVALUATE THE PROBABILITY INTEGERAL AS PER
C     ABRAMOWITZ AND STEGUM EQUATION 26.2.23
C     
      ierr=0
      if ((P.LE.0.5D0).and.(logout.gt.0)) THEN
         write(logout,* )' FSTMJ-E P must be > 0.5'
      endif
      IF (P.LE.0.5D0) THEN
         ierr=1
         return
      endif
      T=1.0D0-P
      T=SQRT(DLOG(1.0D0/(T*T)))
cd      write(logout,* )' FSTMJ-I  T    =',t
      Xp = t - ((0.010328D0*t + 0.802853D0)*t + 2.515517D0) /
     *     (((0.001308D0*t + 0.189269D0)*t + 1.432788D0)*t + 1.0D0)
cd      write(logout,* )' FSTMJ-I  Xp   =',Xp
c
c end of equation 26.2.23
c
      if((n.lt.1).and.(logout.gt.0)) then
         write(logout,* )' FSTMJ-F  n must be greater than zero',n
      endif
      if (n.lt.1) then
         ierr=1
         return
      endif
      V1=N
      if((m.lt.1).and.(logout.gt.0)) then
         write(logout,* )' FSTMJ-F m must be greater than zero',m
      endif
      if (m.lt.1) then
         ierr=1
         return
      endif
      V2=M
      if((n.lt.5).or.(m.lt.5)) go to 2000
c branch on case of low number of degrees of freedom
c
c now use A&S 26.6.16 to approximate F(n,m,1-p)
c
      a=v2*0.5D0
cD      write(logout,* )' FSTMJ-I  a    =',a
      b=v1*0.5D0
cD      write(logout,* )' FSTMJ-I  b    =',b
      z1=1.0D0/(2.0D0*a-1.0D0)
      z2=1.0D0/(2.0D0*b-1.0D0)
      h=2.0D0/(z1+z2)
cD      write(logout,* )' FSTMJ-I  h    =',h
      lambda=(xp*xp-3.0D0)/6.0D0
cD      write(logout,* )' FSTMJ-I lambda=',lambda
      w=Xp*sqrt(h+lambda)/h - (z2-z1)*
     *     (lambda+0.8333333333D0-0.66666667D0/h)
cD      write(logout,* )' FSTMJ-I w     =',w
      FSTMJ=dexp(2.0D0*w)
      return
 2000 continue
c here for special case of low number of degrees  of freedom 
c do a numerical inverse of of A&S equation 26.6.15
c to approximate F(n,m,1-p)
      fl=1.0D0
      fh=1.0D0
 2100 continue
      fh=fh*2.0D0
      x=fstatz(fh,v1,v2)
cD      write(logout,* )' FSTMJ-I for Fh=',Fh,'    X=',x
      if(dabs(x-xp).lt.0.0001) then
         FSTMJ=FH
         return
      endif
      if(x.lt.xp) then
         fl=fh
         go to 2100
      endif
 2200 continue
      f=(fl+fh)*0.5D0
cD      write(logout,* )' FSTMJ-I  f,fl,fh=',f,fl,fh
      x=fstatz(f,v1,v2)
cD      write(logout,* )' FSTMJ-I        x=',x
      if(dabs(fl-fh).lt.0.0001) then
         FSTMJ=f
         return
      endif
      if(x.lt.xp) then
         fl=f
      else
         fh=f
      endif            
      go to 2200
      end FUNCTION FSTMJ


      real*8 function fstatz(f,v1,v2)
c     
C     THIS ROUTINE CALCULATES THE PROBABILITY OF AN F-STATISTIC WITH V1 AND
C     V2 DEGREES OF FREEDOM
C     
      real*8 f,v1,v2,q,vv1,vv2
      q=dexp(dlog(f)/3.0D0)
      vv1=2.0D0/9.0D0/v1
      vv2=2.0D0/9.0D0/v2
      fstatz = (q*(1.0D0-vv2)-(1.0D0-vv1))/sqrt(vv1+q*q*vv2)
      return
      end function fstatz


      subroutine dfxnum(ans,numparms,itrctl,
     *     x,n,ders,EFX,FXS,IFLAG2,NUMPTS)
C     
C THIS ROUTINE PERFORMS NUMERICAL PARTIAL DERIVATIVES OF THE FITTING
C FUNCTION
c
c declarations of the calling sequence
      integer numparms
      real*8 ans(numparms)
      logical itrctl(numparms)
      real*8 x(numpts),fxs(numpts)
      integer n
      real*8 ders(numparms)
      external EFX
      real*8 EFX
      integer IFLAG2
c     end of calling declarations
c     
      real*8 tmpanswers,delta
      allocatable tmpanswers(:)
      real*8 tmp1,tmp2,tmp3,tmp4,tmp5,del
      logical exist1,exist2,exist4,exist5,shrink !CHANGED 8/8/91
      integer expand,smaller    !ADDED 8/8/91
      integer iparm,i,nloop,IFLAG
c
c
c
      allocate (tmpanswers(numparms))
      tmp3=FXS(N)
      exist1=.false.
      exist2=.false.
      exist4=.false.
      exist5=.false.
      do iparm=1,numparms
c     
c     skip out if not fitting to that parm
         if(.not.itrctl(iparm)) cycle
         shrink=.false.         !ADDED 8/8/91
         expand=0               !ADDED 8/8/91
         smaller=0              !ADDED 8/8/91
         do i=1,numparms
            tmpanswers(i)=ans(i)
         enddo
         delta=Dabs(ans(iparm)*0.001D0)
         if(delta.eq.0.0D0) delta=0.001
         nloop=0
 1100    continue
         nloop=nloop+1
         if(nloop.gt.100) then
            IFLAG2=1
            deallocate (tmpanswers)
            return
         endif      
         tmpanswers(iparm)=ans(iparm)+delta
         IFLAG=0
         TMP4=EFX(TMPANSWERS,X,N,IFLAG)
         if(IFLAG.ne.0) then
            delta=delta/2.
            shrink=.true.       !ADDED 8/8/91
            go to 1100
         endif
c     
c     if function value is Zero then accept delta
         if(tmp3.eq.0.0D0) go to 1150
c     
c     if function values are the same, accept delta
         if(tmp3.eq.tmp4) go to 1150
c     
         del = DABS(tmp4/tmp3-1.0D0)
c     
         if(expand.ge.2) go to 1150 !ADDED 8/8/91
c     !ADDED 8/8/91
c     if already expandED twice accept delta                              !ADDED 8/8/91
c     !ADDED 8/8/91
         if((del.lt.0.0001).and.(.not.shrink)) then !CHANGED 8/8/91
            delta=delta*10.D0
            expand=expand+1     !ADDED 8/8/91
            go to 1100
         endif
         if(smaller.ge.4) go to 1150 !ADDED 8/8/91
c     !ADDED 8/8/91
c     if delta has been decreased by 4 orders of magnitude accept delta   !ADDED 8/8/91
c     !ADDED 8/8/91
         if(del.gt.0.03) then
            delta=delta/10.D0
            smaller=smaller+1   !ADDED 8/8/91
            go to 1100
         endif      
 1150    continue
         exist4=.true.
         tmpanswers(iparm)=ans(iparm)-delta
         IFLAG=0
         TMP2=EFX(TMPANSWERS,X,N,IFLAG)
         if(IFLAG.eq.0)       exist2=.true.
         tmpanswers(iparm)=ans(iparm)-2.0D0*delta
         IFLAG=0
         TMP1=EFX(TMPANSWERS,X,N,IFLAG)
         if(IFLAG.eq.0)       exist1=.true.
         tmpanswers(iparm)=ans(iparm)+2.0D0*delta
         IFLAG=0
         TMP5=EFX(TMPANSWERS,X,N,IFLAG)
         if(IFLAG.eq.0) exist5=.true.
         if(exist1.and.exist2.and.exist4.and.exist5) then
            ders(iparm)=(tmp1-8.0D0*(tmp2-tmp4)-tmp5)/12.0D0/delta            
            cycle
         endif
         if(exist2.and.exist4) then
            ders(iparm)=(tmp4-tmp2)/2.0D0/delta
            cycle
         endif
         ders(iparm)=(tmp4-tmp3)/delta
      enddo
      deallocate (tmpanswers)
      return
      end subroutine dfxnum



      SUBROUTINE TQL2(NM,N,D,E,Z,IERR)
C
      INTEGER I,J,K,L,M,N,II,L1,L2,NM,MML,IERR
      real*8 D(n),E(N),Z(nm,n)
      real*8 C,C2,C3,DL1,EL1,F,G,H,P,R,S,S2,TST1,TST2,PYTHAG
C
C     THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE TQL2,
C     NUM. MATH. 11, 293-306(1968) BY BOWDLER, MARTIN, REINSCH, AND
C     WILKINSON.
C     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 227-240(1971).
C
C     THIS SUBROUTINE FINDS THE EIGENVALUES AND EIGENVECTORS
C     OF A SYMMETRIC TRIDIAGONAL MATRIX BY THE QL METHOD.
C     THE EIGENVECTORS OF A FULL SYMMETRIC MATRIX CAN ALSO
C     BE FOUND IF  TRED2  HAS BEEN USED TO REDUCE THIS
C     FULL MATRIX TO TRIDIAGONAL FORM.
C
C     ON INPUT
C
C        NM MUST BE SET TO THE ROW DIMENSION OF TWO-DIMENSIONAL
C          ARRAY PARAMETERS AS DECLARED IN THE CALLING PROGRAM
C          DIMENSION STATEMENT.
C
C        N IS THE ORDER OF THE MATRIX.
C
C        D CONTAINS THE DIAGONAL ELEMENTS OF THE INPUT MATRIX.
C
C        E CONTAINS THE SUBDIAGONAL ELEMENTS OF THE INPUT MATRIX
C          IN ITS LAST N-1 POSITIONS.  E(1) IS ARBITRARY.
C
C        Z CONTAINS THE TRANSFORMATION MATRIX PRODUCED IN THE
C          REDUCTION BY  TRED2, IF PERFORMED.  IF THE EIGENVECTORS
C          OF THE TRIDIAGONAL MATRIX ARE DESIRED, Z MUST CONTAIN
C          THE IDENTITY MATRIX.
C
C      ON OUTPUT
C
C        D CONTAINS THE EIGENVALUES IN ASCENDING ORDER.  IF AN
C          ERROR EXIT IS MADE, THE EIGENVALUES ARE CORRECT BUT
C          UNORDERED FOR INDICES 1,2,...,IERR-1.
C
C        E HAS BEEN DESTROYED.
C
C        Z CONTAINS ORTHONORMAL EIGENVECTORS OF THE SYMMETRIC
C          TRIDIAGONAL (OR FULL) MATRIX.  IF AN ERROR EXIT IS MADE,
C          Z CONTAINS THE EIGENVECTORS ASSOCIATED WITH THE STORED
C          EIGENVALUES.
C
C        IERR IS SET TO
C          ZERO       FOR NORMAL RETURN,
C          J          IF THE J-TH EIGENVALUE HAS NOT BEEN
C                     DETERMINED AFTER 30 ITERATIONS.
C
C     CALLS PYTHAG FOR  SQRT(A*A + B*B) .
C
C     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO BURTON S. GARBOW,
C     MATHEMATICS AND COMPUTER SCIENCE DIV, ARGONNE NATIONAL LABORATORY
C
C     THIS VERSION DATED AUGUST 1983.
C
C     This version modified for virtual arrays by M.L.Johnson Dec 1985
C
C     ------------------------------------------------------------------
C
      IERR = 0
      IF (N .EQ. 1) GO TO 1001
C     
      DO I = 2, N
         E(I-1) = E(I)
      enddo
C
      F = 0.0D0
      TST1 = 0.0D0
      E(N) = 0.0D0
C     
      DO L = 1, N
         J = 0
         H = DABS(D(L)) + DABS(E(L))
         IF (TST1 .LT. H) TST1 = H
C     .......... LOOK FOR SMALL SUB-DIAGONAL ELEMENT ..........
         DO M = L, N
            TST2 = TST1 + DABS(E(M))
            IF (TST2 .EQ. TST1) GO TO 120
C     .......... E(N) IS ALWAYS ZERO, SO THERE IS NO EXIT
C     THROUGH THE BOTTOM OF THE LOOP ..........
         enddo
C     
 120     IF (M .EQ. L) GO TO 220
 130     IF (J .EQ. 30) GO TO 1000
         J = J + 1
C     .......... FORM SHIFT ..........
         L1 = L + 1
         L2 = L1 + 1
         G = D(L)
         P = (D(L1) - G) / (2.0D0 * E(L))
         R = PYTHAG(P,1.0D0)
         D(L) = E(L) / (P + SIGN(R,P))
         D(L1) = E(L) * (P + SIGN(R,P))
         DL1 = D(L1)
         H = G - D(L)
         IF (L2 .GT. N) GO TO 145
C     
         DO I = L2, N
            D(I) = D(I) - H
         enddo
C
 145     F = F + H
C     .......... QL TRANSFORMATION ..........
         P = D(M)
         C = 1.0D0
         C2 = C
         EL1 = E(L1)
         S = 0.0D0
         MML = M - L
C     .......... FOR I=M-1 STEP -1 UNTIL L DO -- ..........
         DO II = 1, MML
            C3 = C2
            C2 = C
            S2 = S
            I = M - II
            G = C * E(I)
            H = C * P
            R = PYTHAG(P,E(I))
            E(I+1) = S * R
            S = E(I) / R
            C = P / R
            P = C * D(I) - S * G
            D(I+1) = H + S * (C * G + S * D(I))
C     .......... FORM VECTOR ..........
            DO K = 1, N
               H = Z(K,I+1)
               Z(K,I+1) = S * Z(K,I) + C * H
               Z(K,I) = C * Z(K,I) - S * H
            enddo
C
         enddo
C
         P = -S * S2 * C3 * EL1 * E(L) / DL1
         E(L) = S * P
         D(L) = C * P
         TST2 = TST1 + DABS(E(L))
         IF (TST2 .GT. TST1) GO TO 130
  220    D(L) = D(L) + F
      enddo

C     .......... ORDER EIGENVALUES AND EIGENVECTORS ..........
      DO II = 2, N
         I = II - 1
         K = I
         P = D(I)
C     
         DO J = II, N
            IF (D(J) .GE. P) cycle
            K = J
            P = D(J)
         enddo
C
         IF (K .EQ. I) cycle
         D(K) = D(I)
         D(I) = P
C
         DO J = 1, N
            P = Z(J,I)
            Z(J,I) = Z(J,K)
            Z(J,K) = P
         enddo
C
      enddo
C
      GO TO 1001
C     .......... SET ERROR -- NO CONVERGENCE TO AN
C                EIGENVALUE AFTER 30 ITERATIONS ..........
 1000 IERR = L
 1001 RETURN
      END SUBROUTINE TQL2


      real*8 FUNCTION PYTHAG(A,B)
C     
C     FROM EISPAC ROUTINES
C     
      real*8 A,B,mlj
C     
C     FINDS SQRT(A**2+B**2) WITHOUT OVERFLOW OR DESTRUCTIVE UNDERFLOW
C     
      real*8 P,R,S,T,U
      P = DMAX1(DABS(A),DABS(B))
      IF (P .EQ. 0.0D0) GO TO 20
      R = (DMIN1(DABS(A),DABS(B))/P)
      r=r*r
 10   CONTINUE
      T = 4.0D0 + R
      IF (T .EQ. 4.0D0) GO TO 20
      S = R/T
      U = 1.0D0 + 2.0D0*S
      P = U*P
      mlj=S/U
      R = mlj*mlj * R
      GO TO 10
 20   PYTHAG = P
      RETURN
      END FUNCTION PYTHAG


      SUBROUTINE TRED2(NM,N,D,E,Z)
C     
      INTEGER I,J,K,L,N,II,NM,JP1
      real*8 D(n),E(N),Z(nm,n)
      real*8 F,G,H,HH,SCALE
C
C     THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE TRED2,
C     NUM. MATH. 11, 181-195(1968) BY MARTIN, REINSCH, AND WILKINSON.
C     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 212-226(1971).
C     
C     THIS SUBROUTINE REDUCES A REAL SYMMETRIC MATRIX TO A
C     SYMMETRIC TRIDIAGONAL MATRIX USING AND ACCUMULATING
C     ORTHOGONAL SIMILARITY TRANSFORMATIONS.
C     
C     ON INPUT
C     
C     NM MUST BE SET TO THE ROW DIMENSION OF TWO-DIMENSIONAL
C     ARRAY PARAMETERS AS DECLARED IN THE CALLING PROGRAM
C     DIMENSION STATEMENT.
C     
C     N IS THE ORDER OF THE MATRIX.
C     
C     Z CONTAINS THE REAL SYMMETRIC INPUT MATRIX.  ONLY THE
C     LOWER TRIANGLE OF THE MATRIX NEED BE SUPPLIED.
C     
C     ON OUTPUT
C     
C     D CONTAINS THE DIAGONAL ELEMENTS OF THE TRIDIAGONAL MATRIX.
C     
C     E CONTAINS THE SUBDIAGONAL ELEMENTS OF THE TRIDIAGONAL
C     MATRIX IN ITS LAST N-1 POSITIONS.  E(1) IS SET TO ZERO.
C     
C     Z CONTAINS THE ORTHOGONAL TRANSFORMATION MATRIX
C     PRODUCED IN THE REDUCTION.
C     
C     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO BURTON S. GARBOW,
C     MATHEMATICS AND COMPUTER SCIENCE DIV, ARGONNE NATIONAL LABORATORY
C     
C     THIS VERSION DATED AUGUST 1983.
C     
C     This version modified for virtual arrays by M.L.Johnson Dec 1985
C     
C     ------------------------------------------------------------------
C     
      DO I = 1, N
         D(I) = z(N,I)
      enddo
C     
      IF (N .EQ. 1) GO TO 510
C     .......... FOR I=N STEP -1 UNTIL 2 DO -- ..........
      DO II = 2, N
         I = N + 2 - II
         L = I - 1
         H = 0.0D0
         SCALE = 0.0D0
         IF (L .LT. 2) GO TO 130
C     .......... SCALE ROW (ALGOL TOL THEN NOT NEEDED) ..........
         DO K = 1, L
            SCALE = SCALE + DABS(D(K))
         enddo
C     
         IF (SCALE .NE. 0.0D0) GO TO 140
 130     E(I) = D(L)
C     
         DO J = 1, L
            D(J) = Z(L,J)
            Z(I,J) = 0.0D0
            Z(J,I) = 0.0D0
         enddo
C
         GO TO 290
C
  140    DO K = 1, L
            D(K) = D(K) / SCALE
            H = H + D(K) * D(K)
         enddo
C
         F = D(L)
         G = -SIGN(SQRT(H),F)
         E(I) = SCALE * G
         H = H - F * G
         D(L) = F - G
C     .......... FORM A*U ..........
         DO J = 1, L
            E(J) = 0.0D0
         enddo
C
         DO J = 1, L
            F = D(J)
            Z(J,I) = F
            G = E(J) + Z(J,J) * F
            JP1 = J + 1
            IF (L .LT. JP1) GO TO 220
C     
            DO K = JP1, L
               G = G + Z(K,J) * D(K)
               E(K) = E(K) + Z(K,J) * F
            enddo
C     
 220        E(J) = G
         enddo
C     .......... FORM P ..........
         F = 0.0D0
C
         DO J = 1, L
            E(J) = E(J) / H
            F = F + E(J) * D(J)
         enddo
C     
         HH = F / (H + H)
C     .......... FORM Q ..........
         DO J = 1, L
            E(J) = E(J) - HH * D(J)
         enddo
C     .......... FORM REDUCED A ..........
         DO J = 1, L
            F = D(J)
            G = E(J)
C
            DO K = J, L
               Z(K,J) = Z(K,J) - F * E(K) - G * D(K)
            enddo
C
            D(J) = Z(L,J)
            Z(I,J) = 0.0D0
         enddo
C
  290    D(I) = H
      enddo
C     .......... ACCUMULATION OF TRANSFORMATION MATRICES ..........
      DO I = 2, N
         L = I - 1
         Z(N,L) = Z(L,L)
         Z(L,L) = 1.0D0
         H = D(I)
         IF (H .EQ. 0.0D0) GO TO 380
C     
         DO K = 1, L
            D(K) = Z(K,I) / H
         enddo
C     
         DO J = 1, L
            G = 0.0D0
C     
            DO K = 1, L
               G = G + Z(K,I) * Z(K,J)
            enddo
C     
            DO  K = 1, L
               Z(K,J) = Z(K,J) - G * D(K)
            enddo
         enddo
C     
 380     DO K = 1, L
            Z(K,I) = 0.0D0
         enddo
C     
      enddo
C     
 510  DO I = 1, N
         D(I) = Z(N,I)
         Z(N,I) = 0.0D0
      enddo
C     
      Z(N,N) = 1.0D0
      E(1) = 0.0D0
      RETURN
      END SUBROUTINE TRED2


      subroutine working(*)
c#ifdef spindrift
c      character*1 out(4)
c      integer*2 irow,icol
c      logical escape
c      external escape
c      data out/'|','/','-','\'/
c      data nc/0/
cc      save nc,out
c      if(escape()) then
c            call clrkb
c            return 1
c            endif
c      call setattr(3)
c      call getpos(irow,icol)
c      nc=nc+1
c      if(nc.gt.4) nc=1
c      call prints(out(nc))
c      call locate(irow,icol)
c#endif
      return
      end subroutine working


      subroutine e_working
c#ifdef spindrift
c      character*1 out
c      integer*2 irow,icol
c      data out/' '/
c      call setattr(3)
c      call getpos(irow,icol)
c      call prints(out)
c      call locate(irow,icol)
c#endif
      return
      end subroutine e_working


      subroutine pause
c#ifdef spindrift
c      character*7 out(2)
c      integer*2 irow,icol
c      data out/'<pause>','       '/
c      call setattr(3)
c      call getpos(irow,icol)
c      call prints(out(1))
c      read(*,*)
c      irow=irow-1
c      call locate(irow,icol)
c      call prints(out(2))
c      call locate(irow,icol)
c#endif
      return
      end subroutine pause
