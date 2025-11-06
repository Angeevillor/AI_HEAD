      subroutine spider_read(nxin, nyin, nsec, nheadrec, nreclength,
     $  image, header, filename)
      implicit none

      character*60 filename
      real image(nxin, nyin, nsec), header(nreclength/4)
      integer nxin, nyin, nsec, nheadrec, nreclength
      integer i, j, k, ii
      logical iset

      open(1, file=filename, form='unformatted', access='direct',
     $     recl=nreclength, status='old')
      
C     *** read in header
      do i=1, nheadrec
         read(1, REC=i)header
      enddo
      
C     *** read in image data
      ii = nheadrec
      do k=1,nsec
         do j=1,nyin
            ii = ii + 1
            read(1, REC=ii)(image(i,j,k), i=1,nxin)
         enddo
      enddo
      
      close(1)
      
      return
      end subroutine spider_read
