import mrcfile
import numpy as np
import glob
import torch
import torch.nn.functional as F
import pandas as pd
import os
import matplotlib.pyplot as plt


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
    
    grouped =df.groupby("class",observed=False)

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


    series=df['class'].value_counts()

    bars=plt.bar(series.index,series.values)


    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}',
                ha='center', va='bottom')

    plt.title("Diameter_distribution")
    plt.savefig(f"{save_path}/Diameter_distribution.png")
        
    df_describe=pd.DataFrame()

    df_describe["class_lable"]=series.index
    df_describe["class_number"]=series.values

    df_describe.to_csv(f"{save_path}/classes_information.csv")



def get_avg_pw(tube_path,tube_length,pad,step,gpu_id,save_path):

    tube_list=sorted(glob.glob(f"{tube_path}/*.mrc"))
    sum_pw=torch.zeros((pad,pad))

    if gpu_id!=None:

        calculate_device=f"cuda:{gpu_id}"
    
    else:

        calculate_device="cpu"
    
    with open(f"{save_path}/run.out","a+") as f:
    
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
  
              edge_side_mean=torch.mean((stack[:,0]+stack[:,-1])/2)
              edge_top_mean=torch.mean((stack[0]+stack[-1])/2)
  
              edge_mean=(edge_side_mean+edge_top_mean)/2
  
              pad_length=(pad-stack.shape[-1])//2
              pad_width=(pad-stack.shape[-2])//2
  
              pad_stack=F.pad(stack,[pad_length,pad_length,pad_width,pad_width],value=edge_mean)
  
              split_stack=torch.split(pad_stack,30)
  
              for m in range(len(split_stack)):
  
                  pw=abs(torch.fft.fftn(split_stack[m].to(calculate_device),dim=(-2,-1)))**2
                  avg_pw=torch.fft.fftshift(torch.log1p(pw),dim=(-2,-1)).sum(dim=0)
                  sum_pw+=avg_pw.cpu()
          else:
  
              pass
              
          print(f"{j+1}/{len(tube_list)} finished!",flush=True,file=f)
            
    

    sum_pw/=len(tube_list)

    with mrcfile.new(f"{save_path}/avg_pw_{pad}.mrc",overwrite=True) as mrc:

        mrc.set_data(sum_pw.numpy())









