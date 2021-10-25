import numpy as np
import cv2 
import os
from scipy import sparse
from sklearn.preprocessing import normalize
import scipy
import tqdm
from PS import *
class Dataset():
    def __init__(self,root="./DiLiGenT/pmsData/ballPNG"):
        self.img=[]
        self.directions=[]
        self.intensities=[]
        dir_direction=os.path.join(root,"light_directions.txt")
        dir_intensity=os.path.join(root,"light_intensities.txt")
        dir_mask=os.path.join(root,"mask.png")
        file_list=os.path.join(root,"filenames.txt")
        self.directions=np.array(self.load_txt(dir=dir_direction))
        self.intensities=np.array(self.load_txt(dir=dir_intensity))
        self.load_img(root=root,dir=file_list)
        self.H,self.W,_=self.img[0].shape
        self.load_mask(filename=dir_mask)
    def load_txt(self,dir=""):
        txt_list=[]
        with open(dir) as f:
            txt_list=f.read().splitlines()
            for i in range(len(txt_list)):
                txt_list[i]=txt_list[i].split()
                txt_list[i]=[float(x) for x in txt_list[i]]
        return txt_list
    def load_img(self,root="",dir=""):
        with open(dir) as f:
            img_dirs=f.read().splitlines()
        for img_dir in img_dirs:
            img_path=os.path.join(root,img_dir)
            img=cv2.imread(img_path)
            self.img.append(img)
    def load_mask(self, filename=None):
        self.mask=cv2.imread(filename,0)
        mask=self.mask.reshape((-1,1))
        self.foreground_ind=np.where(mask != 0)[0]
        self.background_ind=np.where(mask == 0)[0]
    def normal_imags(self):
        for i in tqdm.tqdm(range(len(self.intensities))):
            intensity=self.intensities[i]
            img=self.img[i]
            h,w,_=img.shape
            for j in range(h):
                for k in range(w):
                    img[j,k]=img[j,k]/intensity

if __name__=="__main__":
    root="./DiLiGenT/pmsData/"
    file_list=os.listdir("./DiLiGenT/pmsData/")
    for i in range(len(file_list)):
        if file_list[i][-3:]!= "PNG":
            print(file_list[i][-3:])
            continue
        if not os.path.exists("./"+file_list[i]):
            os.mkdir("./"+file_list[i])
        img_dir=os.path.join(root,file_list[i])
        save_dir="./"+file_list[i]+"/"+file_list[i]+".obj"
        dataset=Dataset(root=img_dir)
        dataset.normal_imags()
        N=get_normal(dataset.img,dataset.directions,dataset.background_ind)
        # N=normal
        depth=compute_depth(N,dataset.mask)
        save_depth(depth,save_dir)
        N[:,:,0], N[:,:,2] = N[:,:,2], N[:,:,0].copy()
        N = (N + 1.0) / 2.0
        cv2.imwrite("./"+file_list[i]+"/"+file_list[i]+"_normal.png",N*255)
        # cv2.imwrite("./"+file_list[i]+"_depth.png",depth*255)
