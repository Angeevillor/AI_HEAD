!****   himpose.f90
! compiled with pgf90 -byteswapio (for SGI byte structured files on Linux PCs)

      program himpose
      implicit none
      
      character*60 filein, symin, mapout, sigmaout
      character*80 arg
      character*60 comment
      character ichar
      character*83 string
      logical existence, proj2d
      integer j, npar, nzlim, nx, ny, irmin, irmax, dum1, dum2, nz_long
      integer nsec, nheadbytes, n_aniso, nheadrec, nphi, nreclength
      real rmin, rmax, scale, pi, twopi, zstart, deltaz, deltaphi
      real header(25), header2(4096), head(10000)
      real, allocatable :: image1(:), image2(:), dencyl(:)
      real, allocatable :: imageout(:), symcyl(:), proj(:)

!***    input parameters:
!           filein (SPIDER image format density)
!           symin  (SPIDER document format for symmetry)
!           mapout (SPIDER image format density)
!           scale  (sampling in data, A/pixel)
!           rmin   (minimum radius to evaluate in helix, in Angtsroms)
!           rmax   (maximum radius to evaluate in helix, in Angtsroms)

      npar = iargc()
      nz_long = -1 ! Initialized to handle when nz_long not provided

      if(npar < 6) then
         write(*,'(A)')
         write(*,'(A)') 'Usage:'
         write(*,'(A)',advance='no') 'himpose volin symdoc volout '
         write(*,'(A)') 'scale rmin rmax [zlen] [proj]'
         write(*,'(A)') 
         write(*,'(A)') 'Required command line arguments:'
         write(*,'(A)') '  volin  - unsymmetrized input volume'
         write(*,'(A)') '  symdoc - symmetry parameter file'
         write(*,'(A)') '  volout - symmetrized output volume'
         write(*,'(A)') '  scale  - pixel size (Ang)'
         write(*,'(A)') '  rmin   - inner tube diameter (Ang)'
         write(*,'(A)') '  rmax   - outer tube diameter (Ang)'
         write(*,'(A)')
         write(*,'(A)') 'Optional command line argument:'
         write(*,'(A)',advance='no') '  zlen   - tube length (pixels)'
         write(*,'(A)') ' Default: dimension volout same as volin'
         write(*,'(A)',advance='no') '  proj   - write out 2D'
         write(*,'(A)') ' projection of map instead of 2D map'
         write(*,'(A)')
         call exit
      endif

      pi = 3.14159
      twopi = 2.*pi
      proj2d = .false.
      
      do j = 1, npar
         call getarg(j, arg)
         if (j == 1) then
            read (arg,*) filein
         else if (j == 2) then
            read (arg,*) symin
         else if (j == 3) then
            read (arg,*) mapout
         else if (j == 4) then
            read (arg,*) scale
         else if (j == 5) then
            read (arg,*) rmin
         else if (j == 6) then
            read (arg,*) rmax
         else if (j == 7) then
            read (arg,*) nz_long
         else if (j == 8) then
            proj2d = .true.
         end if
      enddo

!***  read phistart, zstart from doc file
      
      open(11, file=symin, status='old')
      
!***  read comment line
      read(11,1111)comment
 1111 format(a)
      
!***  move to end of file
 1    read(11,*,end=99) dum1, dum2, deltaphi, zstart
      go to 1
 99   continue
      
      close(11)
      
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

      if(nz_long < 0) then
         nz_long = ny
      else if (nz_long < nsec) then
         write(*,*)
         write(*,*) 'If provided, nz_long must be >= to nsec'
         write(*,*) 'nz_long = ', nz_long
         write(*,*) 'nsec    = ', nsec
         write(*,*)
         write(*,*) 'Exiting himpose'
         write(*,*)
         call exit
      endif

!     irmin = min radius (in pixels) in cylindrical coordinates
!     irmax = max radius (in pixels) in cylindrical coordinates
!     nphi  = number of samples in phi in cylindrical coordinates
      irmin = nint(rmin/scale)
      irmax = nint(rmax/scale)
      
      nphi = twopi*rmax/scale

!     deltaz is rise per subunit (in pixels)
      deltaz=zstart/scale
      
!     nzlim will be the number of substeps taken
      nzlim=nint(deltaz + 2.0)
      
!     n_aniso will be dimension in z for anisotropic symmetrized volume
      n_aniso = 10 + nzlim*(1 + (nz_long/deltaz))
      
!     Dynamically allocate arrays with correct sizes
      allocate(image1(nx*ny*nz_long))
      allocate(image2(nsec*nx*ny))
      allocate(dencyl((irmax-irmin+1)*nphi*ny))
      allocate(imageout(nx*nx*n_aniso))
      allocate(symcyl((irmax-irmin+1)*nphi*nzlim))
      allocate(proj(nx*nz_long))

      call spider_read(nx, ny, nsec, nheadrec, nreclength, image1,
     $     header2, filein)

      call cart_to_cyl(nx, ny, nsec, image1, image2, dencyl, 
     $     scale, irmin, irmax, nphi)
      
!***  have density in cylindrical coords., now impose symmetry
      call symmetry(dencyl, symcyl, nx, ny, scale, imageout, image1,
     $     deltaphi, zstart, irmin, irmax, nphi, nzlim, n_aniso,
     $     nz_long)
      
      if(proj2d) then
         call make_proj(nx, ny, nz_long, image1, proj)
         call spider_write(nx, nz_long, 1, proj, head, mapout)
      else
         call spider_write(nx, nx, nz_long, image1, head, mapout)
      endif

      deallocate(image1, image2, dencyl, imageout, symcyl, proj)

      end program himpose      
      


!*********************************************
      subroutine make_proj(nx, ny, nz, map, proj)
      implicit none
      
      integer nx, ny, nz, i, j, k
      real proj(nx, nz), map(nx, ny, nz)

      do k=1,nz
         do i=1,nx
            proj(i,k) = 0.0
         enddo
      enddo
      
      do k=1,nz
         do j=1,ny
            do i=1,nx
               proj(i,k) = proj(i,k) + map(i,j,k)
            enddo
         enddo
      enddo
      
      end subroutine make_proj


!*********************************************
      subroutine symmetry(dencyl, symcyl, nx, nz, scale, imageout,
     $     resampout, deltaphi, zstart, irmin, irmax, nphi, nzlim,
     $     n_aniso, nz_long)
      implicit none

      integer nx, nz, nz_long
      integer irmin, irmax, nphi, nzlim, n_aniso
      real z, zdif, zwant, zbase, zratio, xcen, ycen, y, y2, x, x2
      real pi, twopi, r, r2, radlim1, radlim2, phiinc
      real deltaphi, zstart, scale
      real dir, dsum, dsum2, diz, density, dgr, den1, den2
      real deltaz, deltadeltaz
      integer nzstep, nsubl, ny, nsubnew, i, j, k, krep, iphip
      integer nxstep, ir, iz, kz, iphi
      real*4 dencyl(irmin:irmax,nphi,nz), symcyl(irmin:irmax,nphi,nzlim)
      real*4 imageout(nx,nx,n_aniso), resampout(nx,nx,nz_long)

      pi = 3.14159
      twopi = 2.*pi
      dgr = twopi/360.
      
      phiinc = twopi/nphi

      deltaz=zstart/scale
      
!     * deltaz is rise in pixels; we want deltadeltaz to be less than 1 pixel
!     * nzlim will be the number of substeps taken
      
      deltadeltaz=real(deltaz/nzlim)
      
!***  nsubl is the number of subunits to average
      nsubl=nint( (2.*nz)/(3.*deltaz) )
      
!     * zbase is starting position in z (in pixels), to exclude end effects
      
      zbase=(nz-nsubl*deltaz)/2.

!     * initialize symmetrized volume array
      do nzstep=1,nzlim
         do iphi=1,nphi
            do ir=irmin,irmax
               symcyl(ir, iphi, nzstep) = 0.0
            enddo
         enddo
      enddo
      
!******************

      do nzstep=1,nzlim
         do iphi=1,nphi
            do ir=irmin,irmax
               
!***  loop over nsubl subunits
               dsum=0.0
               dsum2=0.0
               
               do k=1,nsubl
                  iphip= iphi + nint( ((k-1)*deltaphi*dgr)/ phiinc )
                  
                  if(iphip > nphi)then
 1                   iphip = iphip - nphi
                     if(iphip > nphi) go to 1
                  end if
                  
                  if(iphip < 1)then
 2                   iphip=iphip + nphi
                     if(iphip < 1) go to 2
                  end if
                  
                  z=(k-1)*deltaz + (nzstep-1)*deltadeltaz + zbase
                  iz=z
                  diz=z-iz
                  
                  if(iz > nz-1)then
                     print *,'iz=',iz,' OUT OF BOUNDS - WILL STOP!'
                     stop
                  end if
                  
!***  have iphip, ir, z: interpolate in z
                  
                  den1=dencyl(ir,iphip,iz)
                  den2=dencyl(ir,iphip,iz+1)
                  density=(1.-diz)*den1 + diz*den2
                  
                  dsum=dsum + density
               enddo
               
!***  now have mean density for this ir, iphi, iz      
               
               symcyl( ir, iphi, nzstep) = dsum/nsubl
               
            enddo
         enddo
      enddo

!***  
!***  now generate symmetrized filament in cartesian coordinates
!***  This filament (imageout) will be non-isometric:
!***  dencyl radial scale was same as original Cartesian sampling
!***  new axial scale/section = deltaz (A)/nzlim
      
      ny=nx

      radlim1=irmin*irmin
      radlim2=irmax*irmax

      xcen=nx/2 + 1
      ycen=ny/2 + 1


!**   generate nsubnew subunits to fill nz_long sections
      nsubnew=(nz_long/deltaz) + 1
   
      do krep=1, nsubnew
         do kz=1,nzlim
            k=(krep-1)*nzlim + kz
            
            do j=1, ny
               y= j - ycen
               y2=y*y
               
               do i=1, nx
                  x= i-xcen
                  x2=x*x
                  
                  r2=x2+y2
                  
                  if(r2 > radlim2) then
                     imageout(i,j,k)=0.0
                     cycle      ! fortran 90 - end iteration early
                  end if
                  
                  if(r2 < radlim1) then
                     imageout(i,j,k)=0.0
                     cycle      ! fortran 90 - end iteration early
                  end if
                  
!**   interpolate in r, phi (exact in z)
                  
                  if(x == 0.0 .and. y == 0.0) then
                     iphip=1
                  else
                     iphip=nint(dgr*(57.29578*atan2(y,x)-
     $                    (krep-1)*deltaphi)/phiinc)
                  end if
                  
                  if(iphip > nphi) then
 11                  iphip=iphip - nphi
                     if(iphip > nphi)go to 11
                  end if
                  
                  if(iphip < 1) then
 12                  iphip=iphip + nphi
                     if(iphip < 1) go to 12
                  end if
                  
                  r=sqrt(r2)
                  
                  ir=r
                  dir=r-ir
                  
!!!!  may want to use bilinear interpolation here, but more difficult due to "wrap-around" in phi
                  
                  den1 = symcyl(ir,iphip,kz)
                  den2 = symcyl(ir + 1,iphip,kz)
                  
                  density=(1.-dir)*den1 + dir*den2
                  
                  if(k > n_aniso) then
                     print *,'error:', k,n_aniso
!     stop
                  else
                     imageout(i, j, k)= density      
                  end if
                  
               enddo
            enddo
         enddo
      enddo


!******* now have non-isometric density: x,y = scale (A/pixel); z = deltaz (A)/nzlim 
!     *  resample in z so it is isometric

!***   deltaz was changed from A to pixels, to change back = scale*deltaz
!****    zratio=scale/(scale*deltaz/nzlim) = nzlim/deltaz

      zratio=real(nzlim/deltaz)
      
      do  k=1,nz_long
         zwant=k*zratio
         
         iz=zwant
         zdif=zwant-iz
         
         if(iz > n_aniso)then
            print *,'iz=',iz
            cycle               ! fortran 90 - end iteration early
         end if
         
         do j=1,ny
            do i=1,nx
               den1=imageout(i,j,iz)
               den2=imageout(i,j,iz+1)
               resampout(i,j,k)=(1.-zdif)*den1 + zdif*den2
            enddo
         enddo
      enddo

      end subroutine symmetry
