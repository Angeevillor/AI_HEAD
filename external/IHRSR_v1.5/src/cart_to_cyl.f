      subroutine cart_to_cyl(nx, ny, nsec, image1, image2, dencyl,
     $     scale, irmin, irmax, nphi)
      implicit none

      integer nx, ny, nz, nsec, irmin, irmax, nphi
      real scale, a, b, del1, dely, deltax, r
      real*4 image1(nx,ny,nsec), image2(nsec,nx,ny)
      real*4 dencyl(irmin:irmax,nphi,ny)
      integer i, j, k, ir, ix, iy, iphi
      real pi, twopi, x, y, xcen, ycen, dgr, phiinc
      real phi, csphi(nphi), snphi(nphi)

      pi = 3.14159
      twopi = 2.*pi
      dgr=180./pi

      phiinc = twopi/nphi
      
!**** rotate by 90 degrees so helical axis will be along z
      do k=1,ny
         do j=1,nx
            do i=1,nsec
               image2(i,j,k) = image1(j,k,i)
            enddo
         enddo
      enddo
      
      xcen=(nsec/2) + 1
      ycen=(nx/2) + 1

!**   convert to cylindrical coords
      do k = 1,ny
         do iphi = 1,nphi
            do ir = irmin,irmax
               dencyl(ir,iphi,k) = 0.0
            enddo
         enddo
      enddo

      do iphi=1,nphi
         phi=(iphi-1)*phiinc
         csphi(iphi) = cos(phi)
         snphi(iphi) = sin(phi)
      enddo

      do k=1,ny
         do iphi=1,nphi
            
            do ir= irmin,irmax
               r = float(ir)
               
               x = xcen + r*csphi(iphi)
               y = ycen + r*snphi(iphi)
               
!**** use bilinear interpolation to find density in old Cartesian coordinates for this r,phi point
               ix=int(x)
               iy=int(y)
               if(ix < 1 .or. ix >= nsec)then
                  print *,'ix=',ix
                  cycle  ! fortran 90 - end iteration early
               end if
               
               if(iy < 1 .or .iy>= nx)then
                  print *,'iy=',iy
                  cycle  ! fortran 90 - end iteration early
               end if
               
               deltax = x-real(ix)
               dely   = y-real(iy)
               del1=1.-deltax
               a=(del1*image2(ix,iy,k)+deltax*image2(ix+1,iy,k))*
     $              (1.-dely)
               b=(del1*image2(ix,iy+1,k)+deltax*image2(ix+1,iy+1,k))*
     $              dely
               
               dencyl(ir,iphi,k)=a + b
               
            enddo
         enddo
      enddo

      end subroutine cart_to_cyl
