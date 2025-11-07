import torch
import torch.nn.functional as F
import mrcfile
import numpy as np
import einops
import matplotlib.pyplot as plt
import os
import glob
import sys
import starfile
import pandas as pd
from alive_progress import alive_bar

sys.path.append(os.environ['AI_HEAD_LIB_PATH'])


from utils.grids import coordinate_grid
from utils.interpolation import sample_image_2d

def transform_matrix2d(angle,rad=True,scale=1):

    if rad:

        pass

    else:

        angle=torch.deg2rad(angle)

    n=angle.shape[0]
    matrix=torch.zeros((n,2,2))
    sin=torch.sin(angle)
    cos=torch.cos(angle)
    matrix[...,0,0]=cos/scale
    matrix[...,0,1]=-sin/scale
    matrix[...,1,0]=sin/scale
    matrix[...,1,1]=cos/scale

    return matrix


def extract_tubes(M_dir,coord_dir,width,save_path,angpix=1):

    M_list=sorted(f"{M_dir}/*.mrc")
    
    coord_list=sorted(glob.glob(f"{coord_dir}/*boxes.txt"))
    name_list=[os.path.basename(i).split("_boxes.txt")[0] for i in coord_list]


    M_list=[f"{M_dir}/{name}.mrc" for name in name_list]
    
    with open(f"{save_path}/run.out","a+") as f:
  
      for j in range(len(coord_list)):
  
          if os.path.exists(M_list[j]):
  
  
              try:
  
                  coord=torch.tensor(np.loadtxt(coord_list[j],delimiter="\t")).float()
              
              except:
              
                  coord=torch.tensor(np.loadtxt(coord_list[j],delimiter=" ")).float()
  
              if coord.shape[0]>=2:
  
                  
                  coord[:,0:2]+=coord[:,2:4]//2
  
                  line_coord=coord[:,:2]
  
                  line_coord=einops.rearrange(line_coord,"(n y) x -> n y x",y=2)
                  vector=line_coord[:,1]-line_coord[:,0]
  
                  angle_sign=-1*torch.sign(vector[:,0])
  
                  center_coord=(line_coord[:,1]+line_coord[:,0])//2
                  length=torch.linalg.norm(vector,dim=-1).int()
  
  
                  angle=torch.arccos(vector[:,1]/length)
                  angle*=angle_sign
  
                  M=einops.rearrange(transform_matrix2d(angle),"n y x -> n 1 1 y x")
                  pic=torch.tensor(mrcfile.read(M_list[j]))
  
                  for i in range(length.shape[0]):
  
                      grid=coordinate_grid((length[i],width),torch.tensor((length[i],width))//2)
  
                      grid=torch.flip(grid,dims=(-1,))
                      unrotate_grid=grid.unsqueeze(-1)
                      rotate_grid=(M[i]@unrotate_grid).squeeze(-1)
                      rotate_grid+=center_coord[i]
  
                      rotate_grid=torch.flip(rotate_grid,dims=(-1,))
  
                      sample=sample_image_2d(pic,rotate_grid)
                      
                      sample=torch.nan_to_num(sample)
  
                      with mrcfile.new(f"{save_path}/{name_list[j]}_helix_{i}.mrc",overwrite=True) as mrc:
  
                          mrc.set_data(sample.numpy())
                          mrc.voxel_size=angpix
  
                  print(f"{j+1}/{len(coord_list)} has been finished!",flush=True,file=f)
              
              else:
  
                  print("Invalid coordinate file",flush=True,file=f)
  
  
          else:
  
              print(f"{M_list[j]} is not existed!",flush=True,file=f)
            


def extract_particles(tube_path,box_size,step,save_path,invert,avg=False):

    tube_list=glob.glob(f"{tube_path}/*.mrc")
    name_list=[os.path.basename(i).replace(".mrc",".mrcs") for i in tube_list]
    
    with open(f"{save_path}/run.out","a+") as f:
    
      for j in range(len(tube_list)):
          i=0
          stack=[]
          tube=torch.tensor(mrcfile.read(tube_list[j])) 
                  
          while True: 
              
              stack_i=tube[i:box_size+i].unsqueeze(0)
  
              particle_width=stack_i.shape[-1]
              particle_length=stack_i.shape[-2]
  
              if particle_length<particle_width:
  
                  break
  
              else:
  
                  stack.append(stack_i)
                  i+=step
  
          if len(stack)>=1:
  
              stack=torch.cat(stack,dim=0)
  
              if invert:
                  
                  stack*=-1
  
              if avg:
  
                  with mrcfile.new(f"{save_path}/{name_list[j]}",overwrite=True) as mrc:
  
                      mrc.set_data(stack.mean(dim=0).numpy())
              
              else:
  
                  with mrcfile.new(f"{save_path}/{name_list[j]}",overwrite=True) as mrc:
  
                      mrc.set_data(stack.numpy())
  
              print(f"{j+1}/{len(tube_list)} has been finished!",flush=True,file=f)
  
          else:
  
              pass


def write_metadata(micrograph_meta,particle_path,apix,save_path):

    df=starfile.read(micrograph_meta)
    df_optics=df["optics"]
    df_M=df["micrographs"]

    particle_list=sorted(glob.glob(f"{particle_path}/*.mrcs"))
    particle_meta=[]

    with alive_bar(
            len(particle_list),
                    title=f"writing particles metadata...", 
                    bar="squares"
                    ) as bar:

        for i in range(len(particle_list)):

            particle_key=os.path.basename(particle_list[i]).split("_helix")[0]
            header=mrcfile.open(particle_list[i],header_only=True).header

            
            nz=int(header.nz)
            nx=int(header.nx)

            df_meta=df_M[df_M["rlnMicrographName"].str.contains(particle_key)]

            repeat_df_meta=pd.concat([df_meta]*nz)

            particle_name=[f"{j:06}@{particle_list[i]}" for j in range(nz)]
            repeat_df_meta["rlnImageName"]=particle_name
            repeat_df_meta["rlnAngleRot"]=0
            repeat_df_meta["rlnAngleTilt"]=90
            repeat_df_meta["rlnAnglePsi"]=0
            repeat_df_meta["rlnCtfBfactor"]=0
            repeat_df_meta["rlnCtfScalefactor"]=1
            repeat_df_meta["rlnPhaseShift"]=0
            particle_meta.append(repeat_df_meta)

            bar()


    df_particles=pd.concat(particle_meta)
    df_optics["rlnImagePixelSize"]=apix
    df_optics["rlnImageSize"]=nx
    df_optics["rlnImageDimensionality"]=2
    df_optics["rlnCtfDataAreCtfPremultiplied"]=1
    dic={"optics":df_optics,"particles":df_particles}


    starfile.write(dic,f"{save_path}/particles.star")