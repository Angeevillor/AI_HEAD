import re
def generate_ihrsr_script(particle_num,angular_step,Cnsym,iter,box_size,search_range,maximum_ring,inplane_angular,out_of_plane_tilt_range,cores,particle_stack,
                    pixel_size,outer_radius,path,inner_radius=0,search_step=0.01):
    n=int(re.findall("\d",Cnsym)[0])
    if n==1:
        has_sym=0
        ref_projection_num=360/angular_step
    else: 
        has_sym=1
        ref_projection_num=360/(angular_step*n)


    particle_stack=particle_stack.split(".spi")[0]


    if out_of_plane_tilt_range==0:

        half_range=1

        duplicate_factor=1

        upper_range=1

    else:

        

        duplicate_factor=2*(out_of_plane_tilt_range/angular_step)+1

        half_range=(duplicate_factor+1)/2

        upper_range=duplicate_factor

    content=''' x31 = %s
 x32 = %s
 x33 = %s
 x34 =  %s
 x35 =  %s
 x36 = %s  
 x37 = %s
 x38 = %s
 x44 =  %s
 ;  x78 = total number of references
 x98=x33*%s
 FR L
  <image_stack> %s
 ;
; x31=number of input images 
; x32 = angular increment for projections (in degrees)
; x33 = number of reference projections [=360/x32 for no point group symmetry, =360/(n*x32) for symmetry]
; x34 = number of cycles
; x35 = image dimension (must be square, nx=ny) 
; x36 = search range (pixels) in AP NQ
; x37 = maximum ring (pixels) in AP NQ
; x38 = 1 for point group symmetry, = 0 for no point group symmetry
; x44 = maximum allowed in-plane angular deviation from 0 or 180 degrees  

MD
VB OFF
md
set mp
%s
;
vm
rm -f angleprojdoc.*
;
x89=0.
;  x61=psi
x61=90.
do lb52 x12=1,%s
; x68 is out of plane tilt
x68=(x12-%s)*x32
do lb51 x11=1,x33
;    x65 is azimuthal rotation angle phi
x65=(x11-1.)*x32
x62=90. + x68
x89 = x89 +1
; need to combine the two sets of eulerian angles (phi, theta, psi):
;  azim. rotation about the axis (0.,x65,0.) and tilt (90,x68,90)
;
x67=90.
x66=90.
x64=0.0
;
sa e,x61,x62,x63
x64,x65,x64
x67,x68,x66
;
sd x89,x63,x62,x61
angleprojdoc
lb51
lb52
;
;
;now iterate, starting with projection of previous 3D reconstruction 
;
do lb50 x77 = 2,%s
;
x76=x77-1
;
;
;   rotate volume so helix axis along y
;
rt 3d
volume{***x76}
_1
(0.,90.,90.)
;
pj 3q
_1
((x35/2.)-1)
(1-x98)
angleprojdoc
refproj@****
;
;
ap sh
refproj@****
(1-x98)
x36
(1,x37,1,1)
angleprojdoc
<image_stack>@******
(1-x31)
*
*
n,n
align_doc{***x77}
;
sd e
align_doc{***x77}
;
;now eliminate bad ones 
x90=0
do lb11 x55=1,x31 
; x61,x62,x63 are euler angles of most similar reference
;x64 = most similar reference
;x65 = image number
;x72=in-plane rotation 
;x73=xshift
;x74=yshift
ud ic,x55,x61,x62,x63,x64,x65,x66,x67,x68,x69,x70,x71,x72,x73,x74,x75
align_doc{***x77}
;now get rid of bad ones
;   first get rid of two largest tilt bins
;if(x64.le.x33)goto lb11
;if(x64.gt.(x33*14))goto lb11
;
;now look at in-plane rotations 
if(x72.gt.360)x72 = x72 -360.
if(x72.lt.0)x72 = 360. + x72
if(x72.lt.(180.-x44))then
if(x72.gt.x44)goto lb11
endif
if(x72.gt.(180.+x44))then
if(x72.lt.(360.-x44))goto lb11
endif
;now we are left with good ones! 
x90=x90+1
;apply shifts 
rt sq
<image_stack>@{******x55}
newboxes@{******x90}
x72
x73,x74
; 
;     we just need to copy parameters from align_doc to fangles
sd x90,x61,x62,x63
fangles{***x77}
LB11
;
;
ud ice
angleprojdoc
;
ud ice
align_doc{***x77}
;
sd e
fangles{***x77}
;
;
; now reconstruct
; x52=number of input images, x88 is number of images used for BP 
ud n,x88,x89
fangles{***x77}
sd e
fangles{***x77}
;
IF(x38.eq.0)THEN
;
bp 3f
newboxes@******
(1-x88)
fangles{***x77}
*
ftzdsk{***x77}
;
ELSE
;
bp 3f
newboxes@******
(1-x88)
fangles{***x77}
angsym
ftzdsk{***x77}
;
ENDIF
;
;
;
;now the hard part, impose helical symmetry on threed***
; phistart, zstart (initial guesses at symmetry) 
;
if(x77.gt.2)THEN
vm
hsearch_lorentz ftzdsk{***x77}.spi symdoc.spi %s  %s  %s   %s   %s
endif
;
vm
himpose ftzdsk{***x77}.spi symdoc.spi volume{***x77}.spi %s  %s %s
;
lb50
en d

'''%(particle_num,angular_step,ref_projection_num,iter,box_size,search_range,maximum_ring,has_sym,inplane_angular,duplicate_factor,particle_stack,cores,upper_range,
     half_range,iter,pixel_size,inner_radius,outer_radius,search_step,search_step,pixel_size,inner_radius,outer_radius)

    with open(f"{path}/ihrsr.spd","w+") as f:
        f.write(content)
def generate_symdoc(path,para):
    twist,rise,N=para[0],para[1],int(para[2])
    with open(f"{path}/symdoc.spi","w+") as f:
        f.write(" ;  symdoc.dat, written by GENERATOR program\n")
        f.write(f"    1 2   {twist}    {rise}\n")
        f.write(f"    2 2   {twist}    {rise}\n")
    if N>1:
        with open(f"{path}/angsym.spi","w+") as f:
            f.write(" ;  angsym.dat, written by GENERATOR program\n")
            for i in range(N):
                f.write(f"    {i+1}   3   0    {i*(360/N)}    0\n")
    else:
        pass

def generate_angsym(path,Cnsym):
    n=int(re.findall("\d",Cnsym)[0])
    with open(f"{path}/angsym.spi","w+") as f:
        f.write(" ;  angsym.dat, written by GENERATOR program\n")
        for i in range(n):
            f.write(f"    {i+1}   3   0    {i*(360/n)}    0\n")

