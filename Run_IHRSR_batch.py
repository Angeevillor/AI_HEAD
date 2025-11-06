#!/usr/bin/env python
import glob
import os
import argparse

parser = argparse.ArgumentParser(description="Batch run SPIDER/IHRSR! \n eg: nohup batch_run_IHRSR.py -p (your_ihrsr_projects_path) -s (your_spider_path) ")
parser.add_argument('--project',"-p",help='set the path of your IHRSR projects')
parser.add_argument('--spider',"-s",help='set the path of your SPIDER')

args = parser.parse_args()
dic=vars(args)


if __name__ =="__main__":


    project_path=dic["project"]
    spider_path=dic["spider"]
    
    all_projects_list=sorted(glob.glob(f"{project_path}/*"))
    
    dirs_list=list(filter(lambda x:os.path.isdir(x),all_projects_list))
    
    for i in range(len(dirs_list)):
    
      print(f"Processing {os.path.basename(dirs_list[i])}")
    
      os.system(f"cd {dirs_list[i]} && nohup {spider_path} spd/spi @ihrsr >ihrsr.log")
      
      print(f"jog {i+1} finished!")
