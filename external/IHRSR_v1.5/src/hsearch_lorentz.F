!****    hsearch.f90
! compiled with pgf90 -byteswapio (for SGI byte structured files on Linux PCs)
!
      program hsearch_lorentz
      implicit none
      
      character*80 filein, outdoc, comment
      character*120 arg
      character ichar
      logical existence
      integer n1, n2, j, npar
      integer irmin, irmax, nheadbytes, nheadrec, nreclength
      integer nx, ny, nsec, nphi
      real rmin, rmax, ddz, ddphi, scale
      real phistart, zstart, pi, twopi
      real header(25), header2(4096)
      real, allocatable ::  image1(:), image2(:), dencyl(:)
      
      npar = iargc()
      
!***  input parameters:
!     filein (SPIDER image format density)
!     outdoc (SPIDER document format for symmetry)
!     scale  (sampling in data, A/pixel)
!     rmin   (minimum radius to evaluate in helix, in Angstroms)
!     rmax   (maximum radius to evaluate in helix, in Angstroms)
!     ddphi (increment in degrees for phi search)
!     ddz   (increment in Angstroms for z search)

      pi = 3.14159
      twopi = 2.*pi
      
      if(npar < 7) then
         write(*,'(A)')
         write(*,'(A)') 'Usage:'
         write(*,'(A)',advance='no') 'hsearch_lorentz volin symdoc '
         write(*,'(A)') 'scale rmin rmax phi_inc z_inc'
         write(*,'(A)') 
         write(*,'(A)') 'Required command line arguments:'
         write(*,'(A)') '  volin   - unsymmetrized input volume'
         write(*,'(A)') '  symdoc  - symmetry parameter file'
         write(*,'(A)') '  scale   - pixel size (Ang)'
         write(*,'(A)') '  rmin    - inner tube diameter (Ang)'
         write(*,'(A)') '  rmax    - outer tube diameter (Ang)'
         write(*,'(A)') '  phi_inc - step size for phi search'
         write(*,'(A)') '  z_inc   - step size for z search'
         write(*,'(A)')
         call exit
      endif

      do j = 1, npar
         call getarg(j, arg)
         if (j == 1) then
            read (arg, *) filein
         else if (j == 2) then
            read (arg, *) outdoc
         else if (j == 3) then
            read (arg, *) scale
         else if (j == 4) then
            read (arg, *) rmin
         else if (j == 5) then
            read (arg, *) rmax
         else if (j == 6) then
            read (arg, *) ddphi
         else if (j == 7) then
            read (arg, *) ddz
         end if
      end do


!***  read phistart, zstart from doc file
      
      open(11,file=outdoc,status='unknown')
      
!***  read comment line from symmetry file
      read(11,1111)comment
 1111 format(a)
      
!***  move to end of file
 1    read(11,*,end=99)n1,n2,phistart,zstart
 1001 format(i5,1x,i1,2f12.5)
      go to 1
      
 99   print *,phistart,zstart
      
      n1=n1+1
      
      open(1, file=filein, form='unformatted', access='direct',
     $     recl=128, status='old')
      
      read(1,REC=1)header
      
      nsec=header(1)
      ny=header(2)
      nx=header(12)
      nheadrec=header(13)
      nheadbytes=header(22)
      nreclength=header(23)

      close(1)

!     irmin = min radius (in pixels) in cylindrical coordinates
!     irmax = max radius (in pixels) in cylindrical coordinates
!     nphi  = number of samples in phi in cylindrical coordinates
      irmin = nint(rmin/scale)
      irmax = nint(rmax/scale)
      
      nphi = twopi*rmax/scale
      
      allocate(image1(nx*ny*nsec))
      allocate(image2(nsec*nx*ny))
      allocate(dencyl((irmax-irmin+1)*nphi*ny))

      call spider_read(nx,ny,nsec,nheadrec,nreclength,image1,
     $     header2,filein)

      call cart_to_cyl(nx, ny, nsec, image1, image2, dencyl,
     $     scale, irmin, irmax, nphi)

!***  have density in cylindrical coords., now search for symmetry
!**   start with previous values
            
      call symmetryphi(dencyl,ny,scale,zstart,phistart,
     $     irmin,irmax,nphi,ddphi)
      
!***  this will return new value of phistart, use this for zstart det. 
      
      print *,'1st pass, phistart=', phistart
      
      call symmetryz(dencyl,ny,scale,zstart,phistart,
     $     irmin,irmax,nphi,ddz)
      
      print *,'1st pass, zstart=',zstart
      
      call symmetryphi(dencyl,ny,scale,zstart,phistart,
     $     irmin,irmax,nphi,ddphi)
      
!***  this will return new value of phistart, use this for zstart det. 
      
      print *,'2nd pass, phistart=', phistart
      
      call symmetryz(dencyl,ny,scale,zstart,phistart,
     $     irmin,irmax,nphi,ddz)
      
      print *,'2nd pass, zstart=',zstart
      
!***  now output these values out to doc file
#ifdef PGI
      backspace(11)
#endif
      write(11,1001)n1,n2,phistart,zstart
      
      close(11)

      end program hsearch_lorentz

      

!*********************************************

      subroutine symmetryphi(dencyl,nz,scale,zstart,phistart,
     $     irmin,irmax,nphi,ddp)
      implicit none

      integer irmin, irmax, nphi, nz, j, k, ir, iphi, iz
      integer jjj, nsubl, nzstep, nzlim, iphip
      real ddp, zstart, phistart, scale, a, b, c, d, dgr
      real density, dsum, diz, pi, phiinc, dsum2, ss, totsum
      real deltaphi, deltaz, deltadeltaz, den1, den2, z
      real zbase, twopi
      real*4 dencyl(irmin:irmax,nphi,nz)
      real*8 x(21),y(21), resmin

!**   for a given symmetry (deltaphi,deltaz) calculate residual
      
!     *       deltaz will have units of pixels
      
!**   zstart is starting value for axial rise/subunit (in Angstroms)
!     *            deltaz is axial rise/subunit (in pixels)
!     density has nz sections, so nz/deltaz possible subunits
!     nsubl is number of subunits to use for search
!     = (2/3)*nz/deltaz ; use 2/3 of the possible subunits for search

      pi = 3.14159
      twopi = 2.*pi
      dgr = twopi/360.
      
      phiinc = twopi/nphi
      
      deltaz=zstart/scale

!     deltaz is rise in pixels; we want deltadeltaz to be less than 1 pixel
      nzlim=nint(deltaz + 2.0)
      deltadeltaz=real(deltaz/nzlim)
      
      nsubl=nint( (2.*nz)/(3.*deltaz) )

!     zbase is starting position in z, to exclude end effects

      zbase=(nz-nsubl*deltaz)/2.

!**   will do 21 iterations (+/- 10 steps in deltaphi)
      
      do jjj=1,21

         deltaphi=phistart+ (jjj-11)*ddp
         x(jjj)=deltaphi
         
         totsum=0.0
         
         do iphi=1,nphi
            do nzstep=1,nzlim
               do  ir=irmin,irmax
                  
!***  loop over central nsubl subunits
                  dsum=0.0
                  dsum2=0.0
                  
                  do k=1,nsubl
                     
                     iphip= iphi + nint( ((k-1)*deltaphi*dgr)/ phiinc)
                     
                     if(iphip > nphi)then
 1                      iphip = iphip - nphi
                        if(iphip > nphi)go to 1
                     end if
                     
                     if(iphip < 1)then
 2                      iphip=iphip + nphi
                        if(iphip < 1)go to 2
                     end if
                     
                     
                     z=(k-1)*deltaz + (nzstep-1)*deltadeltaz + zbase
                     iz=z
                     diz=z-iz
                     
                     if(iz > nz-1)then
                        print *,'iz=',iz,' YOU ARE OUT OF RANGE - STOP'
                        stop
                     end if
                     
!***  have iphip, ir, z: interpolate in z
                     
                     den1=dencyl(ir,iphip,iz)
                     den2=dencyl(ir,iphip,iz+1)
                     density=(1.-diz)*den1 + diz*den2
                     
                     dsum=dsum + density
                     dsum2=dsum2 + density*density
                     
                  enddo
                  
                  ss=dsum2 - (dsum*dsum/nsubl)
                  
                  totsum=totsum+ss
                  y(jjj)=totsum
               enddo
            enddo
         enddo
!     print *,x(jjj),y(jjj)
         
      enddo
      
!*****now find best fit for mean phi
      
      a=y(1)
      b=x(11)
      c=.10
      d=-y(11)/10.
      
!     first try fitting to Lorentzian
      
      call lsctest_lorentz (x, y, a, b, c, d)
      phistart=b
      
!     if this value is outside of search range, just take minimum! 
      
      if(b < x(1).or.b > x(21))then
         resmin=y(1)
         do j = 1,21
            if(y(j) <= resmin) then
               resmin = y(j)
               phistart = x(j)
            end if
         end do
      end if
      
      end subroutine symmetryphi

!******************************************

      subroutine symmetryz (dencyl,nz,scale,zstart,phistart,
     $     irmin,irmax,nphi,ddz)
      implicit none
      
      integer nz, irmin, irmax, nphi, ir, iphip, iphi
      integer j, k, iz, jjj, nsubl, nzstep, nzlim
      real ddz, zstart, phistart, scale, a, b, c, d
      real dsum2, dgr, density, den1, deltaphi, deltaz
      real deltadeltaz, den2, dsum, diz, pi, twopi, zbase
      real phiinc, z, ss, totsum
      real*4 dencyl(irmin:irmax,nphi,nz)
      real*8 x(21),y(21), resmin
      
!**   for a given symmetry (deltaphi,deltaz) calculate residual
      
!     *       deltaz will have units of pixels
      
!     *            deltaz is axial rise/subunit (in pixels)
!     density has nz sections, so nz/deltaz possible subunits
!     nsubl = (2/3)*nz/deltaz ; use 2/3 of the subunits for search
      
      pi = 3.14159
      twopi = 2.*pi
      dgr = twopi/360.
      
      phiinc = twopi/nphi
      
      deltaz=zstart/scale
      
      deltaphi=phistart
      
      do jjj=1,21
         
         x(jjj)=(zstart+ (jjj-11)*ddz)
         
         deltaz=x(jjj)/scale
         
         nzlim=nint(deltaz + 2.0)
         
         deltadeltaz=deltaz/nzlim
         
!     *       zbase is starting position in z, to exclude end effects
         nsubl=nint( (2.*nz)/(3.*deltaz) ) - 1
         
         zbase=(nz-nsubl*deltaz)/2.
         
         totsum=0.0
         
         do nzstep=1,nzlim
            do ir=irmin,irmax
               do iphi=1,nphi
                  
!***  loop over central nsubl subunits
                  dsum=0.0
                  dsum2=0.0
                  
                  do k=1,nsubl
                     
                     iphip= iphi + nint(((k-1)*deltaphi*dgr)/ phiinc)
                     
                     if(iphip > nphi)then
 1                      iphip = iphip - nphi
                        if(iphip > nphi)go to 1
                     end if
                     
                     if(iphip < 1)then
 2                      iphip=iphip + nphi
                        if(iphip < 1)go to 2
                     end if
                     
                     z=(k-1)*deltaz + (nzstep-1)*deltadeltaz + zbase
                     iz=z
                     diz=z-iz
                     
                     if(iz > nz-1)then
                        print *,'iz=',iz
                        stop
                     end if
                     
!***  have iphip, ir, z: interpolate in z
                     
                     den1=dencyl(ir,iphip,iz)
                     den2=dencyl(ir,iphip,iz+1)
                     density=(1.-diz)*den1 + diz*den2
                     
                     dsum=dsum + density
                     dsum2=dsum2 + density*density
                     
                  enddo
                  
                  ss=dsum2 - (dsum*dsum/nsubl)
                  
                  totsum=totsum+ss
                  y(jjj)=totsum
               enddo
            enddo
         enddo
         
!     print *,x(jjj),y(jjj)
      enddo
      
!*****now find best fit for mean deltaz
      
      a=y(1)
      b=x(11)
      c=.10
      d=-y(11)/10.
      
!     fit first to Lorentzian
      call lsctest_lorentz (x, y, a, b, c, d)
      zstart=b
!     print *,'a,b,c,d=',a,b,c,d
      
      if(b < x(1).or.b > x(21))then
         resmin=y(1)
         do j = 1,21
            if(y(j) <= resmin) then
               resmin = y(j)
               zstart = x(j)
            end if
         end do
         
      end if
      
      end subroutine symmetryz
