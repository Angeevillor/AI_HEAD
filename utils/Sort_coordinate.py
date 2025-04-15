import glob
import numpy as np
import pandas as pd
import os
import re
import einops

def sort_coordinate(coord_path,tube_path,save_path):

    df=pd.DataFrame()

    tube_list=sorted(glob.glob(f"{tube_path}/*.mrc"))

    M_list=[os.path.basename(i).split("_helix")[0]+"_boxes.txt" for i in tube_list]
    Tube_ID=[int(re.findall("\d+",i)[-1]) for i in tube_list]
    coord_list=[f"{coord_path}/{i}" for i in M_list]
    unique_coord_list=np.unique(np.array(coord_list))
    name_list=[os.path.basename(i) for i in unique_coord_list]

    df["Micrograph"]=coord_list
    df["Tube_ID"]=Tube_ID
    groups=df.groupby("Micrograph")  

    for i in range(len(groups)):

        data=np.loadtxt(unique_coord_list[i],delimiter="\t")
        g=groups.get_group(unique_coord_list[i])

        Tube_ID=g["Tube_ID"].to_numpy()

        raw_index=einops.repeat(Tube_ID,"n -> (n 2)")*2

        add_factor=einops.repeat(np.array((0,1)),"n -> (a n)",a=len(Tube_ID))

        index=raw_index+add_factor

        filtered_coord=data[index]

        np.savetxt(f"{save_path}/{name_list[i]}",filtered_coord,fmt="%d")
