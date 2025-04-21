#!/usr/bin/env python
import os
import sys
sys.path.append(os.environ["AI_HEAD_PATH"])
sys.path.append(os.environ['AI_HEAD_LIB_PATH'])
import math
import shutil
import cv2
import numpy as np
import pandas as pd
import glob
from utils import *
from utils.ctf.ctf_2d import calculate_ctf
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image,ImageDraw
import re
import starfile
import mrcfile
import torch
import argparse


parser = argparse.ArgumentParser(description="Run AI-HEAD by commandline! \n eg: AI-HEAD_command.py --config  ")
parser.add_argument('--config',help='The config file of AI-HEAD')
parser.add_argument('--log',"-o",help='The log name of save folder')

args = parser.parse_args()
dic=vars(args)


if __name__ =="__main__":

    config_dic={}
    with open(dic["config"]) as f:
        lines=f.readlines()
    for line in lines:
        config_dic[line.strip("\n").split()[0]]=line.strip("\n").split()[1]
    
    if config_dic["command"] == 'Image_information':

        log_name=str(dic["log"])
        image_path=config_dic["Image_path"]
        filter_size=int(config_dic["filter_size"])
        filter_intensity=int(config_dic['filter_intensity'])
        lower_limit=int(config_dic["lower_limit"])
        upper_limit=int(config_dic["upper_limit"])

        if os.path.exists(f"Image_information") == True:
            pass
        else:
            os.mkdir(f"Image_information")
        if os.path.exists(f'./Image_information/{log_name}') == True:
            pass
        else:
            os.mkdir(f'./Image_information/{log_name}')

        image_information.show_center(image_path)
        shutil.move('./Image_information/center.txt', f'./Image_information/{log_name}/center.txt')
        image_information.contour_detect(image_path,filter_size,filter_intensity,lower_limit,upper_limit)
        shutil.move('./Image_information/contour_detect.jpg', f'./Image_information/{log_name}/contour_detect.jpg')
        image_information.show(image_path)
        shutil.move('./Image_information/logfile.txt', f'./Image_information/{log_name}/logfile.txt')
        
    elif config_dic["command"] == 'Solve_FFT':

        log_name=str(dic["log"])

        if os.path.exists("Solve_FFT") == True:
            pass
        else:
            os.mkdir("Solve_FFT")
        if os.path.exists(f'./Solve_FFT/{log_name}') == True:
            pass
        else:
            os.mkdir(f'./Solve_FFT/{log_name}')

        image=cv2.imread(config_dic["Solve_the_FFT_image"])
        b,a= image.shape[0],image.shape[1]
        r1=solve_FFT.get_center(config_dic['The_center_coordinates'])
        r2=solve_FFT.get_center(config_dic['The_region_logfile'])
        region=solve_FFT.get_region(config_dic['The_region_logfile'])
        pattern_check=solve_FFT.check_meridian(solve_FFT.get_center(config_dic['The_center_coordinates']),region) #Determine the coordinate range of the diffracted block passing through the meridian
        layer_line=solve_FFT.get_layerline(config_dic['The_region_logfile']) #Get the range of layerlines
        mask1=solve_FFT.mask(image,region)
        check_mask=solve_FFT.mask(image,pattern_check)
        x=r1[0][0]
        y=r1[0][1]
        df=pd.DataFrame(r2,columns=list("XY"))
        point_left=df[(df["Y"]<y)&(df["X"]<x)].values.tolist()
        point_right=df[(df["Y"]<y)&(df["X"]>x)].values.tolist()
        mu=solve_FFT.get_near(point_left,point_right)
        vector1=solve_FFT.vector(r1,point_left)
        vector2=solve_FFT.vector(r1,point_right)
        pair_list=solve_FFT.pair(vector1,vector2)
        dic_layer_distriution={}
        dic_layer={}
        dic_score={}
        dic_count_len={}
        dic_count={}
        dic_lattice_len={}
        dic_lattice={}
        dic_sym_lattice={}
        dic_sym_lattice_len={}
        dic_point_pair={}
        score=[]
        check=[]
        point_pair_1=solve_FFT.point_pair(r1,pair_list[0][0],pair_list[0][1])
        for i in range(len(pair_list)):
            score.append(i+1)
            lattice_1=solve_FFT.lattice(r1,pair_list[i],a,b)
            sym_lattice_1=solve_FFT.sym_lattice(lattice_1,r1[0],a,b)
            point_pair_1=solve_FFT.point_pair(r1,pair_list[i][0],pair_list[i][1])
            check_lattice1=solve_FFT.check_lattice(mu,lattice_1,x,y)
            count_data=solve_FFT.count(lattice_1,sym_lattice_1,mask1)
            check_layer=solve_FFT.check_layerline(layer_line,count_data)
            check_layer_filter=[x for x in check_layer if x > 0]
            check_data=solve_FFT.count(lattice_1,sym_lattice_1,check_mask)
            flag=solve_FFT.check_point(check_lattice1,mask1)
            if len(pattern_check) <=1:
                if len(check_lattice1)>=1:
                    check.append(0)
                else:
                    check.append(1)
            elif len(pattern_check) >1:
                if len(check_data)<=1 or flag<1:
                    check.append(flag)
                else:
                    check.append(1)
            dic_score[i]=i+1
            dic_layer_distriution[i]=check_layer
            dic_layer[i]=len(check_layer_filter)
            dic_count_len[i]=len(count_data)
            dic_count[i]=count_data
            dic_lattice[i]=lattice_1
            dic_lattice_len[i]=len(lattice_1)
            dic_sym_lattice_len[i]=len(sym_lattice_1)
            dic_sym_lattice[i]=sym_lattice_1
            dic_point_pair[i]=point_pair_1
        results=pd.DataFrame()
        results = results._append(dic_layer,ignore_index=True)
        results = results._append(dic_layer_distriution,ignore_index=True)
        results = results._append(dic_count_len,ignore_index=True)
        results = results._append(dic_count,ignore_index=True)
        results = results._append(dic_lattice,ignore_index=True)
        results = results._append(dic_sym_lattice,ignore_index=True)
        results=  results._append(dic_point_pair,ignore_index=True)
        results.index=["layerline_count","layerline_distribution","count","count_data","lattice","sym_lattice","point_pair"]
        results=results.T
        results['check']=check
        results=results.sort_values(by=["check","layerline_count","count"],axis=0,ascending=False)
        rank_results=results.iloc[0:int(config_dic['top_choices']),:]
        rank_file=rank_results[["layerline_count","layerline_distribution","count","check"]]
        rank_file.index=["score"+str(i+1) for i in range(int(config_dic['top_choices']))]
        rank_file.to_csv("./Solve_FFT/"+str(dic["log"])+"/"+"rank_results.csv")
        rank_results.index=["score"+str(i+1) for i in range(int(config_dic['top_choices']))]
        rank_list=["score"+str(i+1) for i in range(int(config_dic['top_choices']))]
        for i in rank_list:
            img_copy=image.copy()
            for point in rank_results.loc[i,"lattice"]:
                cv2.circle(img_copy, [int(point[0]),int(point[1])], 5, (255, 0, 255), -2)
            for point in rank_results.loc[i,"sym_lattice"]:
                cv2.circle(img_copy, [int(point[0]),int(point[1])], 5, (0, 255, 0), -2)
            for point in rank_results.loc[i,"count_data"]:
                cv2.circle(img_copy, [int(point[0]),int(point[1])], 5, (255, 0, 0), -2)
            cv2.circle(img_copy, [int(rank_results.loc[i,"point_pair"][0][0][0]),int(rank_results.loc[i,"point_pair"][0][0][1])], 8, (0, 0, 255), -2)
            cv2.circle(img_copy, [int(rank_results.loc[i,"point_pair"][1][0][0]),int(rank_results.loc[i,"point_pair"][1][0][1])], 8, (0, 0, 255), -2)
            cv2.circle(img_copy, [int(r1[0][0]),int(r1[0][1])], 5, (0, 0, 255), -2)
            if config_dic["display_option"]=="Yes":
                cv2.namedWindow('FFT_pattern')
                cv2.imshow('FFT_pattern', img_copy)
                cv2.imwrite("./Solve_FFT/"+str(dic["log"])+"/"+i+"("+"{}".format(str(rank_results.loc[i,"count"]))+")"+".jpg",img_copy)
                while cv2.waitKey(100) != 27:  # loop if not get ESC
                    if cv2.getWindowProperty("FFT_pattern", cv2.WND_PROP_VISIBLE) <= 0:
                        break
                cv2.destroyAllWindows
            else:
                cv2.imwrite("./Solve_FFT/"+str(dic["log"])+"/"+i+"("+"{}".format(str(rank_results.loc[i,"count"]))+")"+".jpg",img_copy)
                cv2.waitKey()
                cv2.destroyAllWindows
    
    elif config_dic["command"] == 'Basic_vector_coordinate':

        log_name=str(dic["log"])

        if os.path.exists("Basic_vector_coordinate") == True:
            pass
        else:
            os.mkdir("Basic_vector_coordinate")
        if os.path.exists(f"./Basic_vector_coordinate/{log_name}") == True:
            pass
        else:
            os.mkdir(f"./Basic_vector_coordinate/{log_name}")

        raw_data=config_dic["FFT_information"]
        filename=log_name+".txt"
        nums = [float(n) for n in raw_data.split(",")]
        pixel_num = nums[0]
        pixel_size = nums[1]
        r = nums[2]

        M=config_dic['Original_micrograph']
        option=config_dic["Diffraction_option"]

        mrc_data=basis_vector_coordinate.transform(M)
        magnitude_spectrum=basis_vector_coordinate.readimg(M,option)
        center_x,center_y=basis_vector_coordinate.get_center(M)
        basis_vector_coordinate.run(mrc_data,center_x,center_y,magnitude_spectrum,nums,r,pixel_num,pixel_size,filename)
        try:
            shutil.move(f"./Basic_vector_coordinate/{filename}",
            f"./Basic_vector_coordinate/{log_name}/{filename}")
        except:
            pass
        try:
            shutil.move("./Basic_vector_coordinate/diffraction_pattern.png",
                        f"./Basic_vector_coordinate/{log_name}/diffraction_pattern.png")
            shutil.move("./Basic_vector_coordinate/values.txt",
                        f"./Basic_vector_coordinate/{log_name}/values.txt")
        except:
            pass

    elif config_dic["command"] == 'Helix_parameter':

        log_name=str(dic["log"])

        filename=log_name+"_Results.csv"
        add_filename=log_name+"_addition_"+"Results.csv"

        if os.path.exists("Helix_parameter") == True:
            pass
        else:
            os.mkdir("Helix_parameter")
        
        if os.path.exists(f"./Helix_parameter/{log_name}") == True:
            pass
        else:
            os.mkdir(f"./Helix_parameter/{log_name}")

        arr1 = config_dic["The_first_basis_vector_coordinates"]
        arr2=arr1.split(",")
        arr3= config_dic["The_second_basis_vector_coordinates"]
        arr4=arr3.split(",")
        mat=[arr2,arr4]
        n1 = math.floor(float(mat[0][0]))
        Z1 = float(mat[0][1])
        n2 = math.ceil(float(mat[1][0]))
        Z2 = float(mat[1][1])
        n_min=min(abs(n1),abs(n2))
        list1 = []
        i = 1
        while i <= abs(n1):
            list1.append(-i)
            i += 1
        list2 = []
        j = 1
        while j <= n2:
            list2.append(j)
            j += 1
        C = [[x, y] for x in list1 for y in list2]
        list3 = []
        list4 = [] #N/deltaz
        list5 = [] #[k1,k2]
        list6 = [] #N/P
        list7 = [] #deltaz
        list8 = [] #P
        list9 = [] #deltaphi
        list10 = [] #[Z1,Z2]
        list11=[]
        list12=[]
        for i in C:
            m = i[0]
            n = i[1]
            list3.append(helix_parameter.gcd(m, n))
            D = n * Z1 - m * Z2
            list4.append(D)
            k = helix_parameter.K_calculate(m, n)
            list5.append(k)
        for j in list5:
            k1 = float(j[0])
            k2 = float(j[1])
            K = abs(k1 * Z1 + k2 * Z2)
            list6.append(K)
        for i in range(len(C)):
            n1_neo = C[i][0]
            n2_neo = C[i][1]
            k1_neo=list5[i][0]
            k2_neo=list5[i][1]
            N2=k1_neo*n1_neo+k2_neo*n2_neo
            if N2 >0:
                v1_coff=[n1_neo/N2,n2_neo/N2]
                v3_coff=[k2_neo,-k1_neo]
            else:
                v1_coff=[-n1_neo/N2,-n2_neo/N2]
                v3_coff=[-k2_neo,k1_neo]
            if k1_neo + k2_neo >0:
                v2_coff=[n1_neo/N2,n2_neo/N2]
            else:
                v2_coff=[-n1_neo/N2,-n2_neo/N2]
            list11.append([v1_coff[0],v2_coff[0],v3_coff[0]])
            list12.append([v1_coff[1],v2_coff[1],v3_coff[1]])
        for i in range(len(list3)):
            delta_z = float(list3[i]) / float(list4[i])
            list7.append(float("%.2f" % delta_z))
            P = float(list3[i]) / float(list6[i])
            list8.append(float("%.2f" % P))
            delta_phi = 360 * (float(list3[i]) / float(list4[i])) / (float(list3[i]) / float(list6[i]))
            list9.append(float("%.2f" % delta_phi))
            list10.append(["%.5f" %Z1,"%.5f" %Z2])
        dic={"[n1,n2]":C,"rise":list7,"P":list8,"twist":list9,"N":list3,"k_value":list5,"[Z1,Z2]":list10,"[x1_coff]":list11,"[x2_coff]":list12}
        df=pd.DataFrame(dic)
        df.to_csv(f"./Helix_parameter/{log_name}/"+filename,encoding="utf_8_sig")
        add=[[-x,x] for x in range(1,n_min+1)]
        add_N=[]
        add_D=[]
        add_k=[]
        add_K=[]
        add_deltaz=[]
        add_P=[]
        add_phi=[]
        add_Z=[]
        add_x1_coff=[]
        add_x2_coff=[]
        for i in add:
            m=i[0]
            n=i[1]
            add_N.append(helix_parameter.gcd(m,n))
            D = n * Z1 - m * Z2
            add_D.append(D)
        for j in range(len(add_N)):
            add_k.append([0,1])
            k1,k2=0,1
            K = abs(k1 * Z1 + k2 * Z2)
            add_K.append(K)
        for i in range(len(add_N)):
            add_Z.append(["%.5f" %Z1,"%.5f" %Z2])
            delta_z = float(add_N[i]) / float(add_D[i])
            add_deltaz.append(float("%.2f" % delta_z))
            P = float(add_N[i]) / float(add_K[i])
            add_P.append(float("%.2f" % P))
            delta_phi = 360 * (float(add_N[i]) / float(add_D[i])) / (float(add_N[i]) / float(add_K[i]))
            add_phi.append(float("%.2f" % delta_phi))
        for i in range(len(add)):
            n1_neo = add[i][0]
            n2_neo = add[i][1]
            k1_neo=add_k[i][0]
            k2_neo=add_k[i][1]
            N2=k1_neo*n1_neo+k2_neo*n2_neo
            if N2 >0:
                v1_coff=[n1_neo/N2,n2_neo/N2]
                v3_coff=[k2_neo,-k1_neo]
            else:
                v1_coff=[-n1_neo/N2,-n2_neo/N2]
                v3_coff=[-k2_neo,k1_neo]
            if k1_neo + k2_neo >0:
                v2_coff=[n1_neo/N2,n2_neo/N2]
            else:
                v2_coff=[-n1_neo/N2,-n2_neo/N2]
            add_x1_coff.append([v1_coff[0],v2_coff[0],v3_coff[0]])
            add_x2_coff.append([v1_coff[1],v2_coff[1],v3_coff[1]])
        dic2={"[n1,n2]":add,"rise":add_deltaz,"P":add_P,"twist":add_phi,"N":add_N,"k_value":add_k,"[Z1,Z2]":add_Z,"[x1_coff]":add_x1_coff,"[x2_coff]":add_x2_coff}
        df2=pd.DataFrame(dic2)
        df2.to_csv(f"./Helix_parameter/{log_name}/"+add_filename,encoding="utf_8_sig")
    
    elif config_dic["command"] == 'Extract_particles':

        log_name=str(dic["log"])

        if os.path.exists("Extract_particles") == True:
            pass
        else:
            os.mkdir("Extract_particles")

        if os.path.exists(f'./Extract_particles/{log_name}') == True:
            pass
        else:
            os.mkdir(f'./Extract_particles/{log_name}')

        path=f'./Extract_particles/{log_name}'
        
        img_dir=config_dic["Micrographs"]
        coord_dir=config_dic["Coordinates"]
        box_size=int(config_dic["Box_size"])
        step=int(config_dic["Step"])
        angpix=float(config_dic["Pixel_size"])
        ctfm_path=f"{path}/ctfm_Micrograhs"
        tube_path=f"{path}/Tube"
        particle_path=f"{path}/Particles"


        if config_dic["CTF_multiply"]=="Yes":

            if os.path.exists(ctfm_path):
                
                pass

            else:
                os.mkdir(ctfm_path)
            
            if config_dic["GPU"]!="None":


                GPU_id=int(config_dic["GPU"])
        
                calculate_device=f"cuda:{GPU_id}"
            
            else:

                calculate_device="cpu"

            df=starfile.read(config_dic["Star_file"])

            img_list=[f"{img_dir}/{os.path.basename(i)}" for i in df["micrographs"]["rlnMicrographName"].to_list()]
            img_name=[os.path.basename(i) for i in img_list]

            header=df["optics"]
            Voltage=float(header["rlnVoltage"][0])
            Cs=float(header["rlnSphericalAberration"][0])
            Ac=float(header["rlnAmplitudeContrast"][0])
            apix=float(header["rlnMicrographPixelSize"][0])

            df_data=df['micrographs']


            defocus_data=df_data[["rlnDefocusU","rlnDefocusV"]].to_numpy()
            defocus=(defocus_data[:,0]+defocus_data[:,1])/20000
            Ast=df_data["rlnCtfAstigmatism"].to_numpy()/20000
            Angle=df_data['rlnDefocusAngle'].to_numpy()

            print("Conducting Pre-mutiply CTF to micrographs according the star file...")
            n=10
            for i in range(0,len(img_list),n):
                img_stack=torch.cat([torch.tensor(mrcfile.read(j)).unsqueeze(0) for j in img_list[i:i+n]],dim=0).to(calculate_device)
                img_name_split=img_name[i:i+n]
                M_ctf=calculate_ctf(defocus[i:i+n],Ast[i:i+n],Angle[i:i+n],Voltage,Cs,Ac,0,0,apix,img_stack.shape[-2:],True,False).to(calculate_device)
                f_img=torch.fft.rfftn(img_stack,dim=(-2,-1))
                f_ctfm=f_img*M_ctf
                ctfm_img=torch.real(torch.fft.irfftn(f_ctfm,dim=(-2,-1))).cpu()
                for m in range(len(img_name_split)):
                    with mrcfile.new(f"{ctfm_path}/{img_name_split[m]}",overwrite=True) as mrc:
                        mrc.set_data(ctfm_img[m].numpy())
                        mrc.voxel_size=(apix,apix,0)
                print(f"{i+len(img_name_split)}/{len(img_list)} has been pre-mutiplied!")


            if os.path.exists(tube_path):

                pass
            
            else:
                os.mkdir(tube_path)

            if os.path.exists(particle_path):

                pass
            
            else:
                os.mkdir(particle_path)

            print("Extarcting tubes from the ctfm micrographs...")

            Extract.extract_tubes(ctfm_path,coord_dir,box_size,tube_path,angpix)


            if config_dic["Extract_options"]=="Yes":

                pass

            else:

                print("Cutting tubes into particles...")

                Extract.extract_particles(tube_path,box_size,step,particle_path,True)
        
        else:


            if os.path.exists(tube_path):

                pass
            
            else:
                os.mkdir(tube_path)

            if os.path.exists(particle_path):

                pass
            
            else:
                os.mkdir(particle_path)

            Extract.extract_tubes(img_dir,coord_dir,box_size,tube_path,angpix)

            if config_dic["Extract_options"]=="Yes":

                pass

            else:

                Extract.extract_particles(tube_path,box_size,step,particle_path,True)
    
    elif config_dic["command"] == 'Cut_tubes':

        log_name=str(dic["log"])

        if os.path.exists("Cut_tubes") == True:
            
            pass

        else:
            os.mkdir("Cut_tubes")

        if os.path.exists(f'./Cut_tubes/{log_name}') == True:
            
            pass
        
        else:
            os.mkdir(f'./Cut_tubes/{log_name}')

        path=f'./Cut_tubes/{log_name}'
        
        tube_path=config_dic["Tubes"]
        box_size=int(config_dic["Box_size"])
        step=int(config_dic["Step"])
        angpix=float(config_dic["Pixel_size"])

        if config_dic["Invert_options"]=="Yes":

            invert_option=True
        
        else:

            invert_option=False
        
        if config_dic["Average_options"]=="Yes":

            Average_option=True
        
        else:

            Average_option=False

        particle_path=f"{path}/Particles"

        if os.path.exists(particle_path):

            pass
        
        else:
            os.mkdir(particle_path)


        Extract.extract_particles(tube_path,box_size,step,particle_path,invert_option,Average_option)
    

    elif config_dic["command"]=="Diameter_classification":

        log_name=str(dic["log"])
  
        if os.path.exists("Diameter_classification") == True:
            
            pass
        
        else:
            
            os.mkdir("Diameter_classification")
        
        if os.path.exists(f'./Diameter_classification/{log_name}') == True:
            
            pass
        
        else:
            
            os.mkdir(f'./Diameter_classification/{log_name}')

        tube_path=config_dic["Tube_path"]
        save_path=f'./Diameter_classification/{log_name}'
        n_class=int(config_dic["n_class"])

        Diameter_classification.calculate_diameter(tube_path,save_path)

        diameter_file=f"{save_path}/diameter.csv"
        Diameter_classification.reclassication(diameter_file,save_path,n_class)
    
    elif config_dic["command"]=="Average_power_spectra":

        log_name=str(dic["log"])
  
        if os.path.exists("Average_power_spectra") == True:
            
            pass
        
        else:
            
            os.mkdir("Average_power_spectra")
        
        if os.path.exists(f'./Average_power_spectra/{log_name}') == True:
            
            pass
        
        else:
            
            os.mkdir(f'./Average_power_spectra/{log_name}')

        
        tube_path=config_dic["Tube_path"]
        tube_length=int(config_dic["Cut_length"])
        pad_size=int(config_dic["Pad_size"])
        cut_step=int(config_dic["Cut_step"])
        save_path=f'./Average_power_spectra/{log_name}'


        if config_dic["GPU"]!="None":

            GPU_id=int(config_dic["GPU"])
        
        else:

            GPU_id=None

        
        Diameter_classification.get_avg_pw(tube_path,tube_length,pad_size,cut_step,GPU_id,save_path)
    

    elif config_dic["command"]=="Sorting_coordinates":

        log_name=str(dic["log"])
  
        if os.path.exists("Sorting_coordinates") == True:
            
            pass
        
        else:
            
            os.mkdir("Sorting_coordinates")
        
        if os.path.exists(f'./Sorting_coordinates/{log_name}') == True:
            
            pass
        
        else:
            
            os.mkdir(f'./Sorting_coordinates/{log_name}')

        
        tube_path=config_dic["Tube_path"]
        coord_path=config_dic["Coordinate_path"]
        save_path=f'./Sorting_coordinates/{log_name}'
        
        Sort_coordinate.sort_coordinate(coord_path,tube_path,save_path)
               
        
    elif config_dic["command"] == 'Generate_initial_model':


        log_name=str(dic["log"])
  
        if os.path.exists("Generate_initial") == True:
            pass
        else:
            os.mkdir("Generate_initial")

        if os.path.exists(f'./Generate_initial/{log_name}') == True:
            pass
        else:
            os.mkdir(f'./Generate_initial/{log_name}')

        box_size=int(config_dic['Box_size'])
        radius=int(config_dic["Radius"])
        apix=float(config_dic["Pixel_size"])
        bg=int(config_dic["Background"])

        r=int(radius/apix)

        with open(f"./Generate_initial/{str(dic['log'])}/initial.spd","w+") as f:
            f.write("mo 3 \n")
            f.write(f"{str(dic['log'])} \n")
            f.write(f"({box_size},{box_size},{box_size}) \n")
            f.write("c \n")
            f.write(f"{bg} \n")
            f.write("Z \n")
            f.write(f"({r},{box_size}) \n")
            f.write(f"({box_size//2},{box_size//2}) \n")
            f.write(f"({box_size//2},1) \n")
            f.write("Q \n")
            f.write("end \n")
        
        os.system(f"cd ./Generate_initial/{log_name} && {config_dic['Spider_path']} spd/spi @initial")

    elif config_dic["command"] == 'IHRSR':

        log_name=str(dic["log"])

        if os.path.exists("IHRSR") == True:
            pass
        else:
            os.mkdir("IHRSR")
        if os.path.exists(f'./IHRSR/{log_name}') == True:
            pass
        else:
            os.mkdir(f'./IHRSR/{log_name}')
            
        raw_para=pd.read_csv(config_dic["Helix_parameter"])
        
        particle_set=config_dic["Particle_set"]
        angular_step=float(config_dic["Angular_step"])
        particle_num=int(config_dic["Particle_num"])
        box_size=int(config_dic["Box_size"])
        apix=float(config_dic["Pixel_size"])
        iter=int(config_dic["Iteration"])
        search_range=int(config_dic["Serach_range"])
        max_Serach_range=int(config_dic["Max_Serach_range"])
        inplane_angular=int(config_dic["Inplane_angular"])
        oot=int(config_dic["Out_of_plane_tilt_range"])
        cores=int(config_dic["Cores"])
        inner_radius=float(config_dic["Inner_radius"])
        outer_radius=float(config_dic["Outer_radius"])
        parameter_step=float(config_dic["Parameter_step"])


        para=raw_para[["twist","rise","N"]].to_numpy()
        names=[re.findall("\d+",name) for name in raw_para["[n1,n2]"].to_list()]
        n=para.shape[0]
        for i in range(n):
            path="./IHRSR/{}/{}-{}".format(str(dic["log"]),names[i][0],names[i][1])

            if os.path.exists(path) == True:
                pass
            else:
                os.mkdir(path)
            os.system(f"cp {config_dic['Initial_model']} {path}/volume001.spi ")
            IHRSR.generate_symdoc(path,para[i])
            IHRSR.generate_ihrsr_script(particle_num,angular_step,str(para[i][2]),iter,box_size,search_range,max_Serach_range,inplane_angular,oot,cores,
                                        particle_set,apix,outer_radius,path,inner_radius,parameter_step)
            #os.system(f"cd {path} && {dic['Spider_path']} spd/spi @ihrsr >>ihrsr.log ")

            print(f"{i+1}/{n} has been generated!")