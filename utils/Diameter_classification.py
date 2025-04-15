import mrcfile
import numpy as np
import glob
import torch
import torch.nn.functional as F
import pandas as pd
import os


def calculate_diameter(tube_path,save_path):


    tube_list=sorted(glob.glob(f"{tube_path}/*.mrc"))

    tubes_projs=torch.cat([torch.tensor(mrcfile.read(i)).sum(dim=0).unsqueeze(0) for i in tube_list],dim=0)
    radius=tubes_projs.shape[-1]//2
    

    with mrcfile.new(f"{save_path}/tube_projs.mrc",overwrite=True) as mrc:

        mrc.set_data(tubes_projs.numpy())
        
        

    split_projs=torch.chunk(tubes_projs,2,dim=1)

    peak_left,peak_left_index=torch.max(split_projs[0],dim=-1)
    peak_right,peak_right_index=torch.max(split_projs[1],dim=-1)

    weak_left,weak_left_index=torch.min(split_projs[0],dim=-1)
    weak_right,weak_right_index=torch.min(split_projs[1],dim=-1)

    out_diameter=peak_right_index-peak_left_index+radius
    inner_diameter=weak_right_index-weak_left_index+radius

    df=pd.DataFrame()

    df["tube_name"]=tube_list
    df["inner_diameter"]=inner_diameter
    df["out_diameter"]=out_diameter

    df.to_csv(f"{save_path}/diameter.csv")
    

def reclassication(diameter_file,save_path,n_class=8):

    df=pd.read_csv(diameter_file)

    df["bin"]=pd.cut(df["out_diameter"],bins=n_class)

    df["class"]=pd.cut(df["out_diameter"],bins=n_class,labels=[f"class{i}" for i in range(n_class)])
    
    grouped =df.groupby("class")

    labels=[f"class{i}" for i in range(n_class)]
    
    df.to_csv(f"{save_path}/reclassification.csv")

    for i in range(n_class):
    
        if os.path.exists(f"{save_path}/class{i}"):
        
          pass
          
        else:

          os.mkdir(f"{save_path}/class{i}")
        
        try:

          single_part=grouped.get_group(labels[i])
  
          tube_list=single_part["tube_name"].to_list()
  
          for j in range(len(tube_list)):
  
              os.system(f"ln -s {tube_list[j]} {save_path}/class{i}")
        except:
        
          pass


def get_avg_pw(tube_path,tube_length,pad,step,save_path):

    tube_list=sorted(glob.glob(f"{tube_path}/*.mrc"))
    sum_pw=torch.zeros((pad,pad))
    

    for j in range(len(tube_list)):
        i=0
        stack=[]
        tube=torch.tensor(mrcfile.read(tube_list[j]))    
        
        while True: 
            
            stack_i=tube[i:tube_length+i].unsqueeze(0)
            cut_length=stack_i.shape[1]

            if cut_length<tube_length:

                break

            else:

                stack.append(stack_i)
                i+=step

        if len(stack)>1:

            stack=torch.cat(stack,dim=0)

            pad_length=(pad-stack.shape[-1])//2
            pad_width=(pad-stack.shape[-2])//2

            pad_stack=F.pad(stack,[pad_length,pad_length,pad_width,pad_width])

            split_stack=torch.split(pad_stack,30)

            for m in range(len(split_stack)):

                pw=abs(torch.fft.fftn(split_stack[m].cuda(),dim=(-2,-1)))**2
                avg_pw=torch.fft.fftshift(torch.log(pw),dim=(-2,-1)).sum(dim=0)
                sum_pw+=avg_pw.cpu()
        else:

            pass
            
        print(f"{j+1}/{len(tube_list)} finished!")
            
    

    sum_pw/=len(tube_list)

    with mrcfile.new(f"{save_path}/avg_pw_{pad}.mrc",overwrite=True) as mrc:

        mrc.set_data(sum_pw.numpy())









