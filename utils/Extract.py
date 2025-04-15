import torch
import torch.nn.functional as F
import mrcfile
import numpy as np
import einops
import matplotlib.pyplot as plt
import os
import glob
import sys
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

    for j in range(len(coord_list)):
    
        try:

          coord=torch.tensor(np.loadtxt(coord_list[j],delimiter="\t")).float()
          
        except:
          
          coord=torch.tensor(np.loadtxt(coord_list[j],delimiter=" ")).float()
          
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

            with mrcfile.new(f"{save_path}/{name_list[j]}_helix_{i}.mrc",overwrite=True) as mrc:

                mrc.set_data(sample.numpy())
                mrc.voxel_size=angpix

        print(f"{j+1}/{len(coord_list)} has been finished!")

def extract_particles(tube_path,box_size,step,save_path,invert,avg=False):

    tube_list=glob.glob(f"{tube_path}/*.mrc")
    name_list=[os.path.basename(i) for i in tube_list]
    
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

            print(f"{j+1}/{len(tube_list)} has been finished!")

        else:

            pass
