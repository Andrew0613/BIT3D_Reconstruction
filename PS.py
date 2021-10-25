import numpy as np
import cv2 
import os
from scipy import sparse
from sklearn.preprocessing import normalize
import scipy
import tqdm
        
def get_normal(img,directions,background_ind):
    imgs=[np.mean(x,axis=2) for x in img]
    img=None
    H, W = imgs[0].shape
    for i in tqdm.tqdm(range(len(imgs))):
        if img is None:
            img = imgs[i].reshape((-1, 1))
        else:
            img = np.append(img, imgs[i].reshape((-1, 1)), axis=1)
    
    N = scipy.linalg.lstsq(directions, img.T)[0].T
    N = normalize(N, axis=1)  # normalize to account for diffuse reflectance
    if background_ind is not None:
        for i in range(N.shape[1]):
            N[background_ind, i] = 0
        
    N = np.reshape(N, (H, W, 3))
    return N

    
def compute_depth(N,mask):
    H,W= mask.shape
    # 得到掩膜图像非零值索引（即物体区域的索引）
    obj_h, obj_w = np.where(mask != 0)
    # 得到非零元素的数量
    no_pix = np.size(obj_h)
    # 构建一个矩阵 里面的元素值是掩膜图像索引的值
    index = np.zeros((H, W))
    for idx in range(no_pix):
        index[obj_h[idx], obj_w[idx]] = idx
    M = sparse.lil_matrix((2*no_pix, no_pix))
    v = np.zeros((2*no_pix, 1))

    for idx in tqdm.tqdm(range(no_pix)):
        h = obj_h[idx]
        w = obj_w[idx]
        n_x = N[h,w,0]
        n_y = N[h,w,1]
        n_z = N[h,w,2]+1e-8
        if index[h,w+1] and index[h-1,w]:
            M[2*idx, index[h,w]]=(n_z+1e-8)
            M[2*idx, index[h,w+1]]=-(n_z+1e-8)
            v[2*idx,0]=n_x

            M[2*idx+1, index[h,w]]=(n_z+1e-8)
            M[2*idx+1, index[h-1,w]]=-(n_z+1e-8)
            v[2*idx+1,0]=n_y
        elif index[h-1,w]:
            f = -1
            if index[h, w+f]:
                M[2*idx, index[h, w]] = (n_z+1e-8)
                M[2*idx, index[h, w+f]]= -(n_z+1e-8)
                v[2*idx, 0] = f * n_x 
            M[2*idx+1, index[h, w]] = (n_z+1e-8)
            M[2*idx+1, index[h-1, w]]= -(n_z+1e-8)
            v[2*idx+1, 0] = n_y 
        elif index[h, w+1]:
            f = -1
            if index[h-f, w]:
                M[2*idx, index[h, w]] = (n_z+1e-8)
                M[2*idx, index[h-f, w]]= -(n_z+1e-8)
                v[2*idx, 0] = f * n_y 
            M[2*idx+1, index[h, w]] = (n_z+1e-8)
            M[2*idx+1, index[h, w+1]]= -(n_z+1e-8)
            v[2*idx+1, 0] = n_x 
        else:
            f = -1
            if index[h, w+f]:
                M[2*idx, index[h, w]] = (n_z+1e-8)
                M[2*idx, index[h, w+f]]= -(n_z+1e-8)
                v[2*idx, 0] = f * n_x 
            if index[h-f, w]:
                M[2*idx+1, index[h, w]] = (n_z+1e-8)
                M[2*idx+1, index[h-f, w]]= -(n_z+1e-8)
                v[2*idx+1, 0] = f * n_y 
    A=M.T.dot(M)
    B=M.T.dot(v)
    z=sparse.linalg.spsolve(A,B)
    # z=(z-min(z))/(max(z)-min(z))
    z = z - min(z)
    depth=np.zeros((H,W))
    for idx in range(no_pix):
        # 2D图像中的位置
        h = obj_h[idx]
        w = obj_w[idx]
        depth[h, w] = z[idx]
    return depth
def save_depth(depth,save_dir=""):
    if save_dir is "":
        raise Exception("FilePathNULL")
    h,w=depth.shape
    f = open(save_dir, 'w')
    for i in tqdm.tqdm(range(h)):
        for j in range(w):
            if depth[i, j] > 0:
                seq = 'v' + ' ' + str(float(i)) + ' ' + str(float(j)) + ' ' + str(depth[i, j]) + '\n'
                f.writelines(seq)