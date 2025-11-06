      subroutine spider_write(nxout, nyout, nsec, image, 
     $     header, filename)
      implicit none

      integer nheadrec, n, index, ii, i, j, k
      integer nxout, nyout, nsec
      real fmin, fmax, fmean, fsum, fs2
      real header(nxout), header_full(256)
      real image(nxout,nyout,nsec)
      real sigma, ss
      character*60 filename
      character*83 string
      character*12 sdate
      character*8 stime
      equivalence (header_full(212), sdate)
      equivalence (header_full(215), stime)
      logical file_exist
      
      open(1, file=filename, form='unformatted', access='direct',
     $     recl=nxout*4, status='unknown')
      
!***  number of header records
      if(mod(256,nxout) == 0)then
         nheadrec=256/nxout
      else
         nheadrec=(256/nxout) + 1
      end if
      
!     rss      call date(sdate)
!     rss      call time(stime)
      
!***  determine statistics
      fsum=0.0
      fmin=10000.
      fmax=-10000.
      fs2=0.0
      
      do k = 1, nsec
         do j = 1, nyout
            do i = 1, nxout
               fmin=min(fmin, image(i,j,k))
               fmax=max(fmax, image(i,j,k))
               fsum=fsum + image(i,j,k)
               fs2=fs2 + image(i,j,k)**2
            enddo
         enddo
      enddo
      
      n=nxout*nyout*nsec
      fmean=fsum/n
      ss=fs2 - (fsum*fsum/n)
      sigma=sqrt( ss/n )
      
      print *, ' statistics for new SPIDER file:'
      print *, ' fmin=',fmin
      print *, ' fmax=', fmax
      print *, ' fmean=', fmean
      print *, ' sigma=', sigma
      
      do i=1,211
         header_full(i)=0.0
      enddo
      
      do i=217,256
         header_full(i)=0.0
      enddo
      
ccc      do i=212,216
ccc         header_full(i) = 0.0
ccc      enddo

      header_full(1)=nsec
      header_full(2)=nyout
!***  set word 5 to 3 for 3D volume!
      header_full(5)=3
      header_full(6)=1
      header_full(7)=fmax
      header_full(8)=fmin
      header_full(9)=fmean
      header_full(10)=sigma
      header_full(12)=nxout
      header_full(13)=nheadrec
      header_full(22)=nheadrec*nxout*4
      header_full(23)=nxout*4
      
      do i=1,nheadrec
         do j=1,nxout
            index=(i-1)*nxout + j
            if(index <= 256) then
               header(j)=header_full(index)
            else
               header(j)=0.0
            end if 
         enddo
         write(1, REC=i) header
      enddo
      
      ii = nheadrec
      do k= 1,nsec 
         do j= 1,nyout
            ii = ii + 1
            write(1, REC=ii) (image(i,j,k), i=1,nxout)
         enddo
      enddo
      
      close(1)
      
      end subroutine spider_write
