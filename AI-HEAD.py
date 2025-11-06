#!/usr/bin/env python
import os
import sys
sys.path.append(os.environ["AI_HEAD_PATH"])
sys.path.append(os.environ['AI_HEAD_LIB_PATH'])
import math
import shutil
from gooey import Gooey, GooeyParser
import cv2
import numpy as np
import pandas as pd
import glob
# from utils import image_information,solve_FFT,solve_FFT_refine,basis_vector_coordinate,helix_parameter,helix_simulation,sort_diffraction,Extract,IHRSR
from utils import *
from utils.ctf.ctf_2d import calculate_ctf
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image,ImageDraw
import re
import starfile
import mrcfile
import torch
@Gooey(
    richtext_controls=True,
    program_name="AI-HEAD",
    encoding="utf-8",
    #progress_regex=r"^progress: (?P<current>\d+)/(?P<total>\d+)$",
    #progress_expr="current / total * 100",
    #timing_options={
        #'show_time_remaining': True,
        #'hide_time_remaining_on_complete': True,
    #},
    default_size=(900,700),
    menu=[{
        'name': 'File',
        'items': [{
                'type': 'AboutDialog',
                'menuTitle': 'About',
                'name': 'AI-HEAD',
                'description': 'A program for the determination of helical parameters by diffraction patterns',
                'version': '1.0',
                'copyright': '2023',
            }, {
                'type': 'MessageDialog',
                'menuTitle': 'Information',
                'caption': 'Welcome!',
                'message': 'Any suggestions will make this program better!'
            }, {
                'type': 'Link',
                'menuTitle': 'Visit Our Site',
                'url': 'http://feilab.ibp.ac.cn/'
            }]
        },{
        'name': 'Help',
        'items': [{
            'type': 'Link',
            'menuTitle': 'Documentation',
            'url': 'http://feilab.ibp.ac.cn/'
        }]
    }]

)
def main():
    settings_msg = 'A program for the determination of helical parameters by diffraction patterns'
    parser = GooeyParser(description=settings_msg)
    subs = parser.add_subparsers(help='commands', dest='command')
    Image_parser = subs.add_parser('Image_information')
    Image_parser.add_argument('Image_information', widget="FileChooser", help='Please input the FFT image')
    Image_parser.add_argument('lower limit',
                               help='Please enter the lower limit of edge detect', default=0)
    Image_parser.add_argument('upper limit',
                              help='Please enter the upper limit of edge detect', default=255)
    Image_parser.add_argument('filter intensity',
                              help='Intensity value less than this threshold will be ignored', default=125)
    Image_parser.add_argument('filter size',
                              help='Maximum size less than this threshold will be ignored', default=30)
    Image_parser.add_argument('Log File name',
                              help='Please enter the Log File name')
    FFT_parser = subs.add_parser('Solve_FFT')
    FFT_parser.add_argument('Solve the FFT image', widget="FileChooser",help='Please input the FFT image')
    FFT_parser.add_argument('The center coordinates', widget="FileChooser",help='Please enter the center coordinates of the diffraction pattern')
    FFT_parser.add_argument('The region logfile', widget="FileChooser",help='Please enter the logfile of the region')
    FFT_parser.add_argument('top choices',
                            help="Please enter the number of top choices",default=5)
    FFT_parser.add_argument("display option", metavar='Display the results?',help="If you choose to display the results, select 'Yes'.\nIf not, please select 'No', whicherver the opinion is set, the results could be saved automatically under the save folder",choices=["Yes","No"],
                              default="No")
    FFT_parser.add_argument('Save folder name',
                            help='Please enter the Save folder name')
    Refine_parser=subs.add_parser('Solve_FFT_Refine')
    Refine_parser.add_argument('Solve the FFT image', widget="FileChooser",help='Please input the FFT image')
    Refine_parser.add_argument('The center coordinates', widget="FileChooser",help='Please enter the center coordinates of the diffraction pattern')
    Refine_parser.add_argument('The region logfile', widget="FileChooser",help='Please enter the logfile of the region')
    Refine_parser.add_argument("Algorithm mode", metavar='Feature point detection algorithm',
                              help="Please select the Feature point detection algorithm. 0 is refered to TYPE_5_8, 1 is refered to TYPE_7_12, 2 is refered to TYPE_9_16", choices=["0","1","2"],
                              default="2")
    Refine_parser.add_argument("NonmaxSuppression", metavar='Open or close the NonmaxSuppression',
                              help="0: Close 1:Open Open will decrease the number of feature points ", choices=["0","1"],
                              default="0")
    Refine_parser.add_argument('Algorithm Threshold',
                            help='Please enter the Algorithm Threshold',default=5)
    Refine_parser.add_argument('top choices',
                            help="Please enter the number of top choices",default=5)
    Refine_parser.add_argument('Save folder name',
                            help='Please enter the Save folder name')
    Refine_parser.add_argument("display option", metavar='Display the results?',help="If you choose to display the results, select 'Yes'.\nIf not, please select 'No', whicherver the opinion is set, the results could be saved automatically under the save folder",choices=["Yes","No"],
                              default="No")
    vector_parser = subs.add_parser('Basic_vector_coordinate')
    vector_parser.add_argument('Original micrograph', widget="FileChooser",help="Please input the origin data(.mrc format is recommanded)")
    vector_parser.add_argument("Diffraction option", metavar='Do the FFT transformation?',help="If you entered the original helix image, select 'Yes'.\nIf you input a power spectrum, please select 'No'",choices=["Yes","No"],
                              default="Yes")
    vector_parser.add_argument("FFT information", metavar='Diffraction information',help='Please input in order: pixel number, pixel size, helix radius (unit: angstrom) eg: 4096,1.1,187')
    vector_parser.add_argument('Log File name',help='Please enter the Log File name')
    parameter_parser = subs.add_parser('Helix_parameter')
    parameter_parser.add_argument('The first basis vector coordinates',help='Please enter the first basis vector coordinates (n is negative) eg:-2,6')
    parameter_parser.add_argument('The second basis vector coordinates', help='Please enter the second basis vector coordinates (n is positive).)eg:5,8')
    parameter_parser.add_argument('Save file name',help='Please enter the name of parameter file')
    simulation_parser = subs.add_parser('Helix_simulation')
    simulation_parser.add_argument('Helix parameter',
                            widget="FileChooser", help='Please input the helix parameter file to generate a series of helices')
    simulation_parser.add_argument('Helix radius',
                                   help='Please input the helix radius to generate a series of helices')
    simulation_parser.add_argument('Helix length',
                                   help='Please input the helix length to generate a series of helices',default=1024)
    simulation_parser.add_argument('Subunit size',
                                   help='Please input the size of the subunits of helices', default=10)
    simulation_parser.add_argument('Pixel size',
                                   help='Please input the pix_size of the simulated helices')
    simulation_parser.add_argument('Simulation results name',
                                   help='Please enter the Save folder name')
    diffraction_parser = subs.add_parser('Diffraction_simulation')
    diffraction_parser.add_argument("Diffraction option", metavar='Select the mode of diffraction',
                               help="whether the choice is,You should choose a projection to diffraction first,"
                                    "and then adjust the diffraction pattern to get the diffraction pattern."
                                    "\nIf you select 'single', Only this projection will be diffracted."
                                    "\nIf you select all, the rest of the projection is automatically adjusted and diffracted in the same way",
                               choices=["All","single"],
                               default="single")
    diffraction_parser.add_argument('Single diffraction',
                                   widget="FileChooser",
                                   help='Select a projection generated in "Helix_simulation"')
    diffraction_parser.add_argument('Simulation results name',
                                   help='Please enter the Save folder name')
    sort_parser = subs.add_parser('Diffraction_sorting')
    sort_parser.add_argument('Reference_image',widget="FileChooser",
                                   help='Please input the diffraction pattern as a reference')
    sort_parser.add_argument('Simulation_images_path', widget="DirChooser",
                             help='Please input the path of simulation images')
    sort_parser.add_argument('Diffraction features', widget="FileChooser",
                             help='Please input the parameter file got in "Helix_simulation"')
    sort_parser.add_argument("Diffraction region",
                             help="Please input the range of the simulation lattice",default=2048)
    sort_parser.add_argument("Shift number",
                               help="If it is set to 1,the lattice will shift one unit up and down along the meridian "
                                    "\n based on the rise data to get the lattice.",default=2)
    sort_parser.add_argument('Sort results name',
                                    help='Please enter the Save folder name')
    
    extract_parser=subs.add_parser('Extract_particles')

    extract_parser.add_argument('Star_file',widget="FileChooser",
                                   help='Please input the path of starfile')
    extract_parser.add_argument('Micrographs',widget="DirChooser",
                                   help='Please input the dir of micrographs')
    extract_parser.add_argument('Coordinates',widget="DirChooser",
                                   help='Please input the coordinates of the tubes picked by EMAN2')
    extract_parser.add_argument('CTF_multiply',
                              help='Multiply CTF on micrographs in Fourier space',choices=["Yes","No"],default="No")
    extract_parser.add_argument('Box_size',
                              help='Please input the box size of the particles')
    extract_parser.add_argument('Step',
                              help='Set the step of cutting tubes into particles (in pixel)')
    extract_parser.add_argument('Pixel_size',
                              help='Set the pixel size of the images',default=1)
    extract_parser.add_argument('Extract_options',
                              help='Just extract tubes?',choices=["Yes","No"],default="No")
    extract_parser.add_argument('--GPU',
                              help='Please set the device id of GPU if you want to use gpu-acceleration during CTF-mutiply, only single GPU supported.')
    extract_parser.add_argument('Log File name',
                              help='Please enter the Log File name')


    Cut_parser=subs.add_parser('Cut_tubes')

    Cut_parser.add_argument('Star_file',widget="FileChooser",
                                   help='Please input the path of starfile')

    Cut_parser.add_argument('Tubes',widget="DirChooser",
                                   help='Please input the dir of tubes')
    Cut_parser.add_argument('Box_size',
                              help='Please input the box size of the particles')
    Cut_parser.add_argument('Step',
                              help='Set the step of cutting tubes into particles (in pixel)')
    Cut_parser.add_argument('Pixel_size',
                              help='Set the pixel size of the images',default=1)
    Cut_parser.add_argument('Invert_options',
                              help='Invert the contrast?',choices=["Yes","No"],default="Yes")
    Cut_parser.add_argument('Average_options',
                              help='Average the particles without alignmemnt?',choices=["Yes","No"],default="No")
    Cut_parser.add_argument('Log File name',
                              help='Please enter the Log File name')
    

    Classification_parser=subs.add_parser('Diameter_classification')

    Classification_parser.add_argument('Tube_path',widget="DirChooser",
                                   help='Please input the dir of tubes')
    Classification_parser.add_argument('n_class',
                              help='Please set the number of classes to classify the tubes',default=8)
    Classification_parser.add_argument('Log File name',
                              help='Please enter the Log File name')
    
    Avg_pw_parser=subs.add_parser('Average_power_spectra')

    Avg_pw_parser.add_argument('Tube_path',widget="DirChooser",
                                   help='Please input the dir of tubes')
    Avg_pw_parser.add_argument('Cut_length',
                              help='Please set the length to cut tubes')
    Avg_pw_parser.add_argument('Pad_size',
                              help='Please set the size of the power spectrum',default=4096)
    Avg_pw_parser.add_argument('Cut_step',
                              help='Please set the step to cut the tubes (10 percent of the width size is usually used)')
    Avg_pw_parser.add_argument('--GPU',
                              help='Please set the device id of GPU if you want to use gpu-acceleration during power speactrum averaging, only single GPU supported.')
    Avg_pw_parser.add_argument('Log File name',
                              help='Please enter the Log File name')
    
    Sort_coord_parser=subs.add_parser('Sorting_coordinates')

    Sort_coord_parser.add_argument('Tube_path',widget="DirChooser",
                                   help='Please input the dir of tubes')
    Sort_coord_parser.add_argument('Coordinate_path',widget="DirChooser",
                              help='Please input the dir of coordinagte of tubes')
    Sort_coord_parser.add_argument('Log File name',
                              help='Please enter the Log File name')
    

    Initial_parser = subs.add_parser('Generate_initial_model')

    Initial_parser.add_argument('Spider_path',widget="FileChooser",
                                   help='Please input the path of spider')
    
    Initial_parser.add_argument('Box_size',
                              help='Please input the box size of initial model')
    Initial_parser.add_argument('Pixel_size',
                              help='Please input the pixel size of initial model')
    Initial_parser.add_argument('Radius',
                              help='Please input the radius of initial model')
    Initial_parser.add_argument('Background',
                              help='Set the density of background',default=0)
    Initial_parser.add_argument('Log File name',
                              help='Please enter the Log File name')
    
    IHRSR_parser = subs.add_parser('IHRSR')

    IHRSR_parser.add_argument('Spider_path',widget="FileChooser",
                                   help='Please input the path of spider')
    IHRSR_parser.add_argument('Initial_model',widget="FileChooser",
                                   help='Please input the initial model')
    IHRSR_parser.add_argument('Helix_parameter',widget="FileChooser",
                                   help='Please input the helix_parameter calaulated before')
    IHRSR_parser.add_argument('Particle_set',widget="FileChooser",
                                   help='Please input the particles')
    IHRSR_parser.add_argument('Particle_num',
                              help='Set the number of iterations')
    IHRSR_parser.add_argument('Pixel_size',
                              help='Please input the pixel size of initial model')
    IHRSR_parser.add_argument('Box_size',
                              help='Please input the pixel size of initial model')                           
    IHRSR_parser.add_argument('Angular_step',
                              help='Please input the angular step for generating the projections',default=2)
    IHRSR_parser.add_argument('Iteration',
                              help='Set the number of iterations',default=30)
    IHRSR_parser.add_argument('Serach_range',
                              help='Set the serach range between each resolution',default=5)
    IHRSR_parser.add_argument('Max_Serach_range',
                              help='Set the Max serach range, this value should be smaller than (box_size/2)-serach_range',default=38)
    IHRSR_parser.add_argument('Inplane_angular',
                              help='Set the inplane_angular deviation',default=3)
    IHRSR_parser.add_argument('Out_of_plane_tilt_range',
                              help='Set the search range of the out of plane tilt',default=0)
    IHRSR_parser.add_argument('Cores',
                              help='Set the number of cores to conduct the IHRSR',default=32)
    IHRSR_parser.add_argument('Inner_radius',
                              help='Set the inner radius of helix (in Ang)',default=0)
    IHRSR_parser.add_argument('Outer_radius',
                              help='Set the outer radius of helix (in Ang)')
    IHRSR_parser.add_argument('Parameter_step',
                              help='Set the search step of helical parameter',default=0.01)
    
    IHRSR_parser.add_argument('Log File name',
                              help='Please enter the Log File name')


    args = parser.parse_args()
    dic=vars(args)
    if dic["command"] == 'Image_information':
        print(dic)

        if os.path.exists("Image_information") == True:
            pass
        else:
            os.mkdir("Image_information")
        if os.path.exists('./Image_information/{}'.format(str(dic['Log File name']))) == True:
            pass
        else:
            os.mkdir('./Image_information/{}'.format(str(dic['Log File name'])))

        config_keys=["command","Image_path","lower_limit","upper_limit","filter_intensity","filter_size","Log_File_name"]
        values=[dic[key] for key in dic.keys()]

        with open('./Image_information/{}/image_information_config.txt'.format(str(dic['Log File name'])),"w+") as f:

            for i in range(len(config_keys)):

                f.write(f"{config_keys[i]} {values[i]}\n")

        image_information.show_center(dic["Image_information"])
        shutil.move('./Image_information/center.txt', './Image_information/{}'.format(str(dic['Log File name']))+"/center.txt")
        image_information.contour_detect(dic["Image_information"],int(dic["filter size"]),int(dic['filter intensity']),int(dic["lower limit"]),int(dic["upper limit"]))
        shutil.move('./Image_information/contour_detect.jpg', './Image_information/{}'.format(str(dic['Log File name']))+"/contour_detect.jpg")
        image_information.show(dic["Image_information"])
        shutil.move('./Image_information/logfile.txt', './Image_information/{}'.format(str(dic['Log File name']))+"/logfile.txt")

        
    elif dic["command"] == 'Solve_FFT':
        print(dic)
        if os.path.exists("Solve_FFT") == True:
            pass
        else:
            os.mkdir("Solve_FFT")
        if os.path.exists('./Solve_FFT/{}'.format(str(dic['Save folder name']))) == True:
            pass
        else:
            os.mkdir('./Solve_FFT/{}'.format(str(dic['Save folder name'])))

        config_keys=["command","Solve_the_FFT_image","The_center_coordinates","The_region_logfile","top_choices","display_option","Log_File_name"]
        values=[dic[key] for key in dic.keys()]

        with open('./Solve_FFT/{}/solve_fft_config.txt'.format(str(dic['Save folder name'])),"w+") as f:

            for i in range(len(config_keys)):

                f.write(f"{config_keys[i]} {values[i]}\n")

        image=cv2.imread(dic["Solve the FFT image"])
        b,a= image.shape[0],image.shape[1]
        r1=solve_FFT.get_center(dic['The center coordinates'])
        r2=solve_FFT.get_center(dic['The region logfile'])
        region=solve_FFT.get_region(dic['The region logfile'])
        pattern_check=solve_FFT.check_meridian(solve_FFT.get_center(dic['The center coordinates']),region) #Determine the coordinate range of the diffracted block passing through the meridian
        layer_line=solve_FFT.get_layerline(dic['The region logfile']) #Get the range of layerlines
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
        rank_results=results.iloc[0:int(dic['top choices']),:]
        rank_file=rank_results[["layerline_count","layerline_distribution","count","check"]]
        rank_file.index=["score"+str(i+1) for i in range(int(dic['top choices']))]
        rank_file.to_csv("./Solve_FFT/"+str(dic['Save folder name'])+"/"+"rank_results.csv")
        rank_results.index=["score"+str(i+1) for i in range(int(dic['top choices']))]
        rank_list=["score"+str(i+1) for i in range(int(dic['top choices']))]
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
            if dic["display option"]=="Yes":
                cv2.namedWindow('FFT_pattern')
                cv2.imshow('FFT_pattern', img_copy)
                cv2.imwrite("./Solve_FFT/"+str(dic['Save folder name'])+"/"+i+"("+"{}".format(str(rank_results.loc[i,"count"]))+")"+".jpg",img_copy)
                while cv2.waitKey(100) != 27:  # loop if not get ESC
                    if cv2.getWindowProperty("FFT_pattern", cv2.WND_PROP_VISIBLE) <= 0:
                        break
                cv2.destroyAllWindows
            else:
                cv2.imwrite("./Solve_FFT/"+str(dic['Save folder name'])+"/"+i+"("+"{}".format(str(rank_results.loc[i,"count"]))+")"+".jpg",img_copy)
                cv2.waitKey()
                cv2.destroyAllWindows


    elif dic["command"] == 'Solve_FFT_Refine':
        print(dic)
        if os.path.exists("Solve_FFT_Refine") == True:
            pass
        else:
            os.mkdir("Solve_FFT_Refine")
        if os.path.exists('./Solve_FFT_Refine/{}'.format(str(dic['Save folder name']))) == True:
            pass
        else:
            os.mkdir('./Solve_FFT_Refine/{}'.format(str(dic['Save folder name'])))
        image=cv2.imread(dic["Solve the FFT image"])
        b,a= image.shape[0],image.shape[1]
        r1=solve_FFT.get_center(dic['The center coordinates'])
        kp=solve_FFT_refine.fast_detect(dic['Solve the FFT image'],int(dic["Algorithm Threshold"]),int(dic["Algorithm mode"]),int(dic["NonmaxSuppression"]))
        solve_FFT_refine.fast_show(dic['Solve the FFT image'],int(dic["Algorithm Threshold"]),kp,int(dic["Algorithm mode"]),int(dic["NonmaxSuppression"]))
        solve_FFT_refine.show(dic['Solve the FFT image'])
        region=solve_FFT.get_region(dic['The region logfile'])
        shutil.move("./Solve_FFT_Refine/logfile.txt", './Solve_FFT_Refine/{}'.format(str(dic['Save folder name']))+"/logfile.txt")
        select_region=solve_FFT.get_region('./Solve_FFT_Refine/{}'.format(str(dic['Save folder name']))+"/logfile.txt")
        mask1=solve_FFT.mask(image,region)
        mask2=solve_FFT.mask(image,select_region)
        x=r1[0][0]
        y=r1[0][1]
        point_list=solve_FFT.filter(kp,mask2)
        df=pd.DataFrame(point_list,columns=list("XY"))
        point_left=df[(df["Y"]<y)&(df["X"]<x)].values.tolist()
        point_right=df[(df["Y"]<y)&(df["X"]>x)].values.tolist()
        print("The number of results is",len(point_left),"*",len(point_right))
        vector1=solve_FFT.vector(r1,point_left)
        vector2=solve_FFT.vector(r1,point_right)
        pair_list=solve_FFT.pair(vector1,vector2)
        dic_count_len={}
        dic_count={}
        dic_lattice={}
        dic_sym_lattice={}
        dic_point_pair={}
        point_pair_1=solve_FFT.point_pair(r1,pair_list[0][0],pair_list[0][1])
        for i in range(len(pair_list)):
            lattice_1=solve_FFT.lattice(r1,pair_list[i],a,b)
            sym_lattice_1=solve_FFT.sym_lattice(lattice_1,r1[0],a,b)
            point_pair_1=solve_FFT.point_pair(r1,pair_list[i][0],pair_list[i][1])
            count_data=solve_FFT.count(lattice_1,sym_lattice_1,mask1)
            dic_count_len[i]=len(count_data)
            dic_count[i]=count_data
            dic_lattice[i]=lattice_1
            dic_sym_lattice[i]=sym_lattice_1
            dic_point_pair[i]=point_pair_1
        results=pd.DataFrame()
        results = results._append(dic_count_len,ignore_index=True)
        results = results._append(dic_count,ignore_index=True)
        results = results._append(dic_lattice,ignore_index=True)
        results = results._append(dic_sym_lattice,ignore_index=True)
        results=  results._append(dic_point_pair,ignore_index=True)
        results.index=["rank","count_data","lattice","sym_lattice","point_pair"]
        results=results.T
        results_sort=results.sort_values(by="rank",axis=0,ascending=False)
        rank_results=results_sort.iloc[0:int(dic['top choices']),:]
        rank_results.index=["rank"+str(i+1) for i in range(int(dic['top choices']))]
        rank_list=["rank"+str(i+1) for i in range(int(dic['top choices']))]
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
            if dic["display option"]=="Yes":
                cv2.namedWindow('FFT_pattern')
                cv2.imshow('FFT_pattern', img_copy)
                cv2.imwrite("./Solve_FFT_Refine/"+str(dic['Save folder name'])+"/"+i+"("+"{}".format(str(rank_results.loc[i,"rank"]))+")"+".jpg",img_copy)
                while cv2.waitKey(100) != 27:  # loop if not get ESC
                    if cv2.getWindowProperty("FFT_pattern", cv2.WND_PROP_VISIBLE) <= 0:
                        break
                cv2.destroyAllWindows
            else:
                cv2.imwrite("./Solve_FFT_Refine/"+str(dic['Save folder name'])+"/"+i+"("+"{}".format(str(rank_results.loc[i,"rank"]))+")"+".jpg",img_copy)
                cv2.waitKey()
                cv2.destroyAllWindows


    elif dic["command"] == 'Basic_vector_coordinate':
        print(dic)
        if os.path.exists("Basic_vector_coordinate") == True:
            pass
        else:
            os.mkdir("Basic_vector_coordinate")
        if os.path.exists("./Basic_vector_coordinate/{}".format(str(dic["Log File name"]))) == True:
            pass
        else:
            os.mkdir("./Basic_vector_coordinate/{}".format(str(dic["Log File name"])))

        config_keys=["command","Original_micrograph","Diffraction_option","FFT_information","Log_File_name"]
        values=[dic[key] for key in dic.keys()]

        with open("./Basic_vector_coordinate/{}/basic_vector_coordinate_config.txt".format(str(dic["Log File name"])),"w+") as f:

            for i in range(len(config_keys)):

                f.write(f"{config_keys[i]} {values[i]}\n")

        raw_data=dic["FFT information"]
        filename=str(dic["Log File name"])+".txt"
        nums = [float(n) for n in raw_data.split(",")]
        pixel_num = nums[0]
        pixel_size = nums[1]
        r = nums[2]
        mrc_data=basis_vector_coordinate.transform(dic['Original micrograph'])
        magnitude_spectrum=basis_vector_coordinate.readimg(dic['Original micrograph'],dic["Diffraction option"])
        center_x,center_y=basis_vector_coordinate.get_center(dic['Original micrograph'])
        basis_vector_coordinate.run(mrc_data,center_x,center_y,magnitude_spectrum,nums,r,pixel_num,pixel_size,filename)
        try:
            shutil.move("./Basic_vector_coordinate/"+filename,
            './Basic_vector_coordinate/{}'.format(str(dic["Log File name"]))+"/"+ filename)
        except:
            pass
        try:
            shutil.move("./Basic_vector_coordinate/diffraction_pattern.png",
                        './Basic_vector_coordinate/{}'.format(str(dic["Log File name"])) + "/diffraction_pattern.png")
            shutil.move("./Basic_vector_coordinate/values.txt",
                        './Basic_vector_coordinate/{}'.format(str(dic["Log File name"])) + "/values.txt")
        except:
            pass
    elif dic["command"] == 'Helix_parameter':
        print(dic)
        filename=str(dic["Save file name"])+"_Results.csv"
        add_filename=str(dic["Save file name"])+"_addition_"+"Results.csv"
        if os.path.exists("Helix_parameter") == True:
            pass
        else:
            os.mkdir("Helix_parameter")
        
        if os.path.exists("Helix_parameter/{}/".format(str(dic["Save file name"]))) == True:
            pass
        else:
            os.mkdir("Helix_parameter/{}/".format(str(dic["Save file name"])))
        
        save_path="Helix_parameter/{}/".format(str(dic["Save file name"]))

        config_keys=["command","The_first_basis_vector_coordinates","The_second_basis_vector_coordinates","Log_File_name"]
        values=[dic[key] for key in dic.keys()]

        with open("./Helix_parameter/{}/helix_parameter_config.txt".format(str(dic["Save file name"])),"w+") as f:

            for i in range(len(config_keys)):

                f.write(f"{config_keys[i]} {values[i]}\n")
        


        arr1 = dic["The first basis vector coordinates"]
        arr2=arr1.split(",")
        arr3= dic["The second basis vector coordinates"]
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
        df.to_csv(save_path+filename,encoding="utf_8_sig")
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
        df2.to_csv(save_path+add_filename,encoding="utf_8_sig")
        
    elif dic["command"] == 'Helix_simulation':
        print(dic)
        if os.path.exists("Helix_simulation") == True:
            pass
        else:
            os.mkdir("Helix_simulation")
        if os.path.exists("./Helix_simulation/"+str(dic['Simulation results name'])) == True:
            pass
        else:
            os.mkdir("./Helix_simulation/"+str(dic['Simulation results name']))
        if os.path.exists('./Helix_simulation/'+str(dic['Simulation results name'])+"/projection") == True:
            pass
        else:
            os.mkdir("./Helix_simulation/"+str(dic['Simulation results name'])+"/projection")
        data=pd.read_csv(dic["Helix parameter"])
        r=float(dic['Helix radius'])
        l = float(dic['Helix length'])
        P_value=data["P"].to_list()
        twist=data["twist"].to_list()
        delta_z_value=data["rise"].to_list()
        N_value=data["N"].to_list()
        sub_size=int(dic["Subunit size"])
        pixel_size=float(dic["Pixel size"])
        for i in range(len(delta_z_value)):
            helix_simulation.helix_generator(int(l),r,P_value[i],delta_z_value[i],N_value[i],sub_size,pixel_size,"./Helix_simulation/{}/{}/projection{}.png".format(str(dic['Simulation results name']),"projection",str(i+1)))
            with open("./Helix_simulation/{}/{}/parameter.txt".format(str(dic['Simulation results name']),"projection"),"a") as f:
                print('Image and values saved.')
                Z1=N_value[i]*l*pixel_size/P_value[i]
                Z2=l*pixel_size/delta_z_value[i]
                R_n1 = basis_vector_coordinate.get_bessel_order(N_value[i])[int(N_value[i])]
                # print(R_n1)
                X1=(R_n1*l*pixel_size)/(2*math.pi*r)
                print("projection{}:".format(str(i+1)),"{0}, {1}, {2}, {3}, {4}, {5}".format(P_value[i],delta_z_value[i],twist[i],Z1,Z2,X1),file=f, flush=True)
            f.close()


    elif dic["command"] == 'Diffraction_simulation':
        print(dic)
        if os.path.exists("Diffraction_simulation") == True:
            pass
        else:
            os.mkdir("Diffraction_simulation")
        if os.path.exists("./Diffraction_simulation/" + str(dic['Simulation results name'])) == True:
            pass
        else:
            os.mkdir("./Diffraction_simulation/" + str(dic['Simulation results name']))
        if dic["Diffraction option"] =="single":
            path=str(dic["Single diffraction"])
            filename=path.split(os.sep)[-1]
            name=filename.split(".")[0]+".png"
            img=helix_simulation.get_img(dic["Single diffraction"])
            magnitude_spectrum= helix_simulation.FFT(dic["Single diffraction"])
            helix_simulation.run(img,magnitude_spectrum)
            shutil.move('./Diffraction_simulation/values.txt',
                        "./Diffraction_simulation/{}".format(str(dic['Simulation results name'])) + "/values.txt")
            shutil.move("./Diffraction_simulation/test_sim.png",
                        "./Diffraction_simulation/{}".format(str(dic['Simulation results name'])) + "/"+name)
        else:
            file_list=glob.glob("./Helix_simulation/{}/{}/*".format(str(dic['Simulation results name']),"projection"))
            img_list=[file for file in file_list if not file.endswith (".txt")]
            path="./Helix_simulation/"+str(dic['Simulation results name'])+"/projection"
            file_name=os.listdir(path)
            img_name=[file for file in file_name if not file.endswith (".txt")]
            value_dic={}
            with open("./Diffraction_simulation/{}".format(str(dic['Simulation results name'])) + "/values.txt",'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip("\n")
                    key = line.split(":")[0]
                    try:
                        value = float(line.split(":")[1])
                    except:
                        value= []
                        value_raw = line.split(":")[1]
                        value_raw = value_raw.split(",")
                        for i in value_raw:
                            i.strip()
                            value.append(int(i))
                    value_dic[key] = value
            f.close()
            for j in range(len(img_list)):
                img = helix_simulation.get_img(img_list[j])
                magnitude_spectrum = helix_simulation.FFT(img_list[j])
                helix_simulation.batch(img,magnitude_spectrum,value_dic["Zoom"],value_dic["Contrast"],value_dic["Brightness"],value_dic["region"],img_name[j])
                try:
                    shutil.move("./Diffraction_simulation/"+img_name[j],
                            "./Diffraction_simulation/{}".format(str(dic['Simulation results name'])) + "/" + img_name[j])
                except:
                    name=img_name[j].split(".")[0]+".png"
                    shutil.move("./Diffraction_simulation/" + name,
                                "./Diffraction_simulation/{}".format(str(dic['Simulation results name'])) + "/" +
                                name)
    elif dic["command"] == 'Diffraction_sorting':
        print(dic)
        if os.path.exists("Diffraction_sorting") == True:
            pass
        else:
            os.mkdir("Diffraction_sorting")
        if os.path.exists("./Diffraction_sorting/" + str(dic['Sort results name'])) == True:
            pass
        else:
            os.mkdir("./Diffraction_sorting/" + str(dic['Sort results name']))
        if os.path.exists("./Diffraction_sorting/" + str(dic['Sort results name'])+"/converge") == True:
            pass
        else:
            os.mkdir("./Diffraction_sorting/" + str(dic['Sort results name']+"/converge"))
        if os.path.exists("./Diffraction_sorting/" + str(dic['Sort results name'])+"/compare") == True:
            pass
        else:
            os.mkdir("./Diffraction_sorting/" + str(dic['Sort results name']+"/compare"))
        ref=cv2.imread(dic["Reference_image"],0)
        ref_c=Image.fromarray(ref)
        img_list=glob.glob(dic["Simulation_images_path"]+"/*.png")
        path=dic["Simulation_images_path"]
        img_name=[name for name in os.listdir(path) if name.endswith('.png')]
        info_dic={}
        with open(dic["Diffraction features"],'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip("\n")
                key = line.split(":")[0]
                value = []
                value_raw = line.split(":")[1]
                value_raw = value_raw.split(",")
                for i in value_raw:
                    i.strip()
                    value.append(float(i))
                info_dic[key] = value
        for i in range(len(img_list)):
            img=cv2.imread(img_list[i],0)
            img_c=Image.fromarray(img)
            converge_img=sort_diffraction.converge(img_c,ref_c)
            converge_img.save("./Diffraction_sorting/" + str(dic['Sort results name'])+"/converge/converge_"+img_name[i])
        converge_list=glob.glob("./Diffraction_sorting/" + str(dic['Sort results name'])+"/converge/*.png")
        filename=[os.path.basename(f) for f in converge_list]
        file_dict = {os.path.basename(f): f for f in converge_list}
        filename.sort(key=sort_diffraction.extract_number)
        converge_list_sort=[file_dict[f] for f in filename]
        for j in range(len(converge_list)):
            img = Image.open(converge_list_sort[j])
            a,b=img.size
            center_x,center_y=a//2,b//2
            data_list = info_dic["projection{}".format(j + 1)]
            P_value=data_list[0]
            rise=data_list[1]
            twist=data_list[2]
            X=data_list[-1]
            PZ=data_list[-3]
            Z=data_list[-2]
            point_dic_2=sort_diffraction.shift_lattice(img,int(dic["Diffraction region"]),X,PZ,Z,int(dic["Shift number"]))
            print(point_dic_2)
            num=point_dic_2["step"]
            n=point_dic_2["shift"]
            size = 3
            draw = ImageDraw.Draw(img)
            if num >0:
                for i in range(-num,num+1):
                    for k in range(-n,n+1):
                        try:
                            p1=point_dic_2["coordinate_order({0})_pattern({1})".format(i,k)]
                            p2=[2*center_x-p1[0],p1[1]]
                            color={-4:(210,105,30),-3:(255,0,255),-2:(255,255,0),-1:(255,0,0),
                                   0:(0,255,0),
                                   1:(0,0,255),2:(0,255,255),3:(160,32,240),4:(255,127,80)}
                            draw.ellipse((p1[0] - size, p1[1] - size, p1[0] + size, p1[1] + size), fill=color[k])
                            draw.ellipse((p2[0] - size, p2[1] - size, p2[0] + size, p2[1] + size), fill=color[k])
                        except:
                            pass
                        img.save("./Diffraction_sorting/" + str(dic['Sort results name'])+"/compare/compare{}.png".format(str(j+1)))
                with open("./Diffraction_sorting/" + str(dic['Sort results name'])+"/compare/information.txt", 'a', encoding='utf-8') as f:
                            print("compare{}:".format(str(j + 1)),"parameter:{0},{1}".format(rise,twist),file=f,flush=True)
                            print("compare{}:".format(str(j + 1)),"points:{0}".format(point_dic_2),file=f,flush=True)
                            print("\n",file=f,flush=True)

            else:
                pass

    elif dic["command"] == 'Extract_particles':

        print(dic)
        if os.path.exists("Extract_particles") == True:
            pass
        else:
            os.mkdir("Extract_particles")
        if os.path.exists('./Extract_particles/{}'.format(str(dic['Log File name']))) == True:
            pass
        else:
            os.mkdir('./Extract_particles/{}'.format(str(dic['Log File name'])))

        config_keys=["command","Star_file","Micrographs","Coordinates","CTF_multiply",
                     "Box_size","Step","Pixel_size","Extract_options","GPU","Log_File_name"]
        values=[dic[key] for key in dic.keys()]

        with open('./Extract_particles/{}/extract_config.txt'.format(str(dic['Log File name'])),"w+") as f:

            for i in range(len(config_keys)):

                f.write(f"{config_keys[i]} {values[i]}\n")

        path='./Extract_particles/{}'.format(str(dic['Log File name']))
        
        img_dir=dic["Micrographs"]
        coord_dir=dic["Coordinates"]
        box_size=int(dic["Box_size"])
        step=int(dic["Step"])
        angpix=float(dic["Pixel_size"])
        ctfm_path=f"{path}/ctfm_Micrograhs"
        tube_path=f"{path}/Tube"
        particle_path=f"{path}/Particles"

        if dic["CTF_multiply"]=="Yes":

            if os.path.exists(ctfm_path):
                
                pass

            else:
                os.mkdir(ctfm_path)


            if dic["GPU"]!=None:

                GPU_id=int(dic["GPU"])
        
                calculate_device=f"cuda:{GPU_id}"
            
            else:

                calculate_device="cpu"

            df=starfile.read(dic["Star_file"])

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
                # f_ctfm=f_img*torch.sign(M_ctf)
                f_ctfm=f_img*M_ctf
                ctfm_img=torch.fft.irfftn(f_ctfm,dim=(-2,-1)).cpu()

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


            if dic["Extract_options"]=="Yes":

                pass

            else:

                print("Cutting tubes into particles...")

                Extract.extract_particles(tube_path,box_size,step,particle_path,invert=True)

                Extract.write_metadata(dic["Star_file"],particle_path,angpix,particle_path)
        
        else:


            if os.path.exists(tube_path):

                pass
            
            else:
                os.mkdir(tube_path)

            if os.path.exists(particle_path):

                pass
            
            else:
                os.mkdir(particle_path)

            print("Extarcting tubes from the micrographs...")

            Extract.extract_tubes(img_dir,coord_dir,box_size,tube_path,angpix)

            if dic["Extract_options"]=="Yes":

                pass

            else:

                Extract.extract_particles(tube_path,box_size,step,particle_path,invert=True)

                Extract.write_metadata(dic["Star_file"],particle_path,angpix,particle_path)
    
    elif dic["command"] == 'Cut_tubes':

        print(dic)
        if os.path.exists("Cut_tubes") == True:
            pass
        else:
            os.mkdir("Cut_tubes")

        if os.path.exists('./Cut_tubes/{}'.format(str(dic['Log File name']))) == True:
            pass
        else:
            os.mkdir('./Cut_tubes/{}'.format(str(dic['Log File name'])))
        
        config_keys=["command","Star_file","Tubes","Box_size","Step","Pixel_size","Invert_options","Average_options","Log_File_name"]
        values=[dic[key] for key in dic.keys()]

        with open('./Cut_tubes/{}/Cut_config.txt'.format(str(dic['Log File name'])),"w+") as f:

            for i in range(len(config_keys)):

                f.write(f"{config_keys[i]} {values[i]}\n")

        path='./Cut_tubes/{}'.format(str(dic['Log File name']))
        
        tube_path=dic["Tubes"]
        box_size=int(dic["Box_size"])
        step=int(dic["Step"])
        angpix=float(dic["Pixel_size"])


        if dic["Invert_options"]=="Yes":

            invert_option=True
        
        else:

            invert_option=False
        
        if dic["Average_options"]=="Yes":

            Average_option=True
        
        else:

            Average_option=False

        particle_path=f"{path}/Particles"

        if os.path.exists(particle_path):

            pass
        
        else:
            os.mkdir(particle_path)


        Extract.extract_particles(tube_path,box_size,step,particle_path,invert_option,Average_option)

        Extract.write_metadata(dic["Star_file"],particle_path,angpix,particle_path)


    elif dic["command"]=="Diameter_classification":

        print(dic)

        log_name=str(dic['Log File name'])
  
        if os.path.exists("Diameter_classification") == True:
            
            pass
        
        else:
            
            os.mkdir("Diameter_classification")
        
        if os.path.exists(f'./Diameter_classification/{log_name}') == True:
            
            pass
        
        else:
            
            os.mkdir(f'./Diameter_classification/{log_name}')
        
        config_keys=["command","Tube_path","n_class","Log_File_name"]
        values=[dic[key] for key in dic.keys()]
        
        with open(f'./Diameter_classification/{log_name}/diameter_classification_config.txt',"w+") as f:

            for i in range(len(config_keys)):

                f.write(f"{config_keys[i]} {values[i]}\n")

        tube_path=dic["Tube_path"]
        save_path=f'./Diameter_classification/{log_name}'
        n_class=int(dic["n_class"])

        Diameter_classification.calculate_diameter(tube_path,save_path)

        diameter_file=f"{save_path}/diameter.csv"
        Diameter_classification.reclassication(diameter_file,save_path,n_class)
    
    elif dic["command"]=="Average_power_spectra":

        log_name=str(dic['Log File name'])
  
        if os.path.exists("Average_power_spectra") == True:
            
            pass
        
        else:
            
            os.mkdir("Average_power_spectra")
        
        if os.path.exists(f'./Average_power_spectra/{log_name}') == True:
            
            pass
        
        else:
            
            os.mkdir(f'./Average_power_spectra/{log_name}')
        
        config_keys=["command","Tube_path","Cut_length","Pad_size","Cut_step","GPU","Log_File_name"]
        values=[dic[key] for key in dic.keys()]
        
        with open(f'./Average_power_spectra/{log_name}/average_power_spectra_config.txt',"w+") as f:

            for i in range(len(config_keys)):

                f.write(f"{config_keys[i]} {values[i]}\n")

        
        tube_path=dic["Tube_path"]
        tube_length=int(dic["Cut_length"])
        pad_size=int(dic["Pad_size"])
        cut_step=int(dic["Cut_step"])
        save_path=f'./Average_power_spectra/{log_name}'

        if dic["GPU"]!=None:

            GPU_id=int(dic["GPU"])
        
        else:

            GPU_id=dic["GPU"]

        
        Diameter_classification.get_avg_pw(tube_path,tube_length,pad_size,cut_step,GPU_id,save_path)
    

    elif dic["command"]=="Sorting_coordinates":

        log_name=str(dic['Log File name'])
  
        if os.path.exists("Sorting_coordinates") == True:
            
            pass
        
        else:
            
            os.mkdir("Sorting_coordinates")
        
        if os.path.exists(f'./Sorting_coordinates/{log_name}') == True:
            
            pass
        
        else:
            
            os.mkdir(f'./Sorting_coordinates/{log_name}')
        
        config_keys=["command","Tube_path","Coordinate_path","Log_File_name"]
        values=[dic[key] for key in dic.keys()]

        with open(f'./Sorting_coordinates/{log_name}/sorting_coordinates_config.txt',"w+") as f:

            for i in range(len(config_keys)):

                f.write(f"{config_keys[i]} {values[i]}\n")

        
        tube_path=dic["Tube_path"]
        coord_path=dic["Coordinate_path"]
        save_path=f'./Sorting_coordinates/{log_name}'
        
        Sort_coordinate.sort_coordinate(coord_path,tube_path,save_path)            
        
        
    elif dic["command"] == 'Generate_initial_model':
        print(dic)
        if os.path.exists("Generate_initial") == True:
            pass
        else:
            os.mkdir("Generate_initial")
        if os.path.exists('./Generate_initial/{}'.format(str(dic['Log File name']))) == True:
            pass
        else:
            os.mkdir('./Generate_initial/{}'.format(str(dic['Log File name'])))

        config_keys=["command","Spider_path","Box_size","Pixel_size","Radius","Background","Log_File_name"]
        values=[dic[key] for key in dic.keys()]

        with open('./Generate_initial/{}/Initial_config.txt'.format(str(dic['Log File name'])),"w+") as f:

            for i in range(len(config_keys)):

                f.write(f"{config_keys[i]} {values[i]}\n")


        box_size=int(dic['Box_size'])
        radius=int(dic["Radius"])
        apix=float(dic["Pixel_size"])
        bg=int(dic["Background"])

        r=int(radius/apix)

        with open(f"./Generate_initial/{str(dic['Log File name'])}/initial.spd","w+") as f:
            f.write("mo 3 \n")
            f.write(f"{str(dic['Log File name'])} \n")
            f.write(f"({box_size},{box_size},{box_size}) \n")
            f.write("c \n")
            f.write(f"{bg} \n")
            f.write("Z \n")
            f.write(f"({r},{box_size}) \n")
            f.write(f"({box_size//2},{box_size//2}) \n")
            f.write(f"({box_size//2},1) \n")
            f.write("Q \n")
            f.write("end \n")
        
        os.system(f"cd ./Generate_initial/{str(dic['Log File name'])} && {dic['Spider_path']} spd/spi @initial")

    elif dic["command"] == 'IHRSR':
        print(dic)
        if os.path.exists("IHRSR") == True:
            pass
        else:
            os.mkdir("IHRSR")
        if os.path.exists('./IHRSR/{}'.format(str(dic['Log File name']))) == True:
            pass
        else:
            os.mkdir('./IHRSR/{}'.format(str(dic['Log File name'])))

        config_keys=["command","Spider_path","Initial_model","Helix_parameter","Particle_set","Particle_num",
                    "Pixel_size","Box_size","Angular_step","Iteration","Serach_range",
                    "Max_Serach_range","Inplane_angular","Out_of_plane_tilt_range","Cores",
                    "Inner_radius","Outer_radius","Parameter_step","Log_File_name"]
        values=[dic[key] for key in dic.keys()]

        with open('./IHRSR/{}/ihrsr_config.txt'.format(str(dic['Log File name'])),"w+") as f:

            for i in range(len(config_keys)):

                f.write(f"{config_keys[i]} {values[i]}\n")

        raw_para=pd.read_csv(dic["Helix_parameter"])
        particle_set=dic["Particle_set"]
        angular_step=float(dic["Angular_step"])
        particle_num=int(dic["Particle_num"])
        box_size=int(dic["Box_size"])
        apix=float(dic["Pixel_size"])
        iter=int(dic["Iteration"])
        search_range=int(dic["Serach_range"])
        max_Serach_range=int(dic["Max_Serach_range"])
        inplane_angular=int(dic["Inplane_angular"])
        oot=int(dic["Out_of_plane_tilt_range"])
        cores=int(dic["Cores"])
        inner_radius=float(dic["Inner_radius"])
        outer_radius=float(dic["Outer_radius"])
        parameter_step=float(dic["Parameter_step"])


        para=raw_para[["twist","rise","N"]].to_numpy()
        names=[re.findall("\d+",name) for name in raw_para["[n1,n2]"].to_list()]
        n=para.shape[0]
        for i in range(n):
            path="./IHRSR/{}/{}-{}".format(str(dic['Log File name']),names[i][0],names[i][1])

            if os.path.exists(path) == True:
                pass
            else:
                os.mkdir(path)
            os.system(f"cp {dic['Initial_model']} {path}/volume001.spi ")
            IHRSR.generate_symdoc(path,para[i])
            IHRSR.generate_ihrsr_script(particle_num,angular_step,str(para[i][2]),iter,box_size,search_range,max_Serach_range,inplane_angular,oot,cores,
                                        particle_set,apix,outer_radius,path,inner_radius,parameter_step)
            #os.system(f"cd {path} && {dic['Spider_path']} spd/spi @ihrsr >>ihrsr.log ")

            print(f"{i+1}/{n} has been generated!")
            

if __name__ == '__main__':
    main()
