#------------------------------------------------------
# Exact diagonalization code for the 2D Hubbard model
#
#                               　　　　　　　　 YN, Ph.D
#                                03/01/2017(mm/dd/yyyy)
#This might have bugs.
#This code is just for studying the ED method.
#
#
#------------------------------------------------------
from scipy.sparse import lil_matrix,csr_matrix
#from scipy.linalg
from scipy.sparse import linalg
from numpy.linalg import norm
import numpy as np
import scipy as sc
import scipy.misc as scm

def calc_matc(isite,nx,ny,p):
    nc = nx*ny
    nf = 4**nc
    mat_c = lil_matrix((nf,nf))

    for jj in range(nf):
        vec_i = calc_ii2vec(jj,nc)
#        print "initial",vec_i,p
        vec_iout,sig = calc_c_cd(isite,vec_i,p,nc)
#        print "sig",sig
        if sig != 0:
            ii = calc_vec2ii(vec_iout,nc)
#            print ii,vec_iout
            mat_c[ii,jj] = sig

                
        
    return mat_c

def calc_matc_fix(isite,nx,ny,nup,ndown,mf,mf2,vec_hi,vec_hi2,p):
    nc = nx*ny
    nf = 4**nc
    mat_c = lil_matrix((mf2,mf))

    for jj in range(nf):
        jh = vec_hi[jj]

        if jh != -1:
            vec_i = calc_ii2vec(jj,nc)
            #        print "initial",vec_i,p
            vec_iout,sig = calc_c_cd(isite,vec_i,p,nc)
            #        print "sig",sig
            if sig != 0:
                ii = calc_vec2ii(vec_iout,nc)
                ih = vec_hi2[ii]
#                print ih
                if ih != -1:
#                    print ii,vec_iout
                    mat_c[ih,jh] = sig
                
                
        
    return mat_c


def calc_vec2ii(vec_iout,nc):
    ii = 0
#    print "iouts",vec_iout
    for isite in range(2*nc):
#        print isite
#        print ii
#        print "iout",vec_iout[isite]
        ii +=  vec_iout[isite]*2**(isite)

    return ii

def calc_c_cd(isite,vec_i,p,nc):
    vec_iout = vec_i
    sig = calc_sign(isite,vec_i,p,nc)
    if sig == 0:
        vec_iout = -1
    else:
        vec_iout[isite] = p
                
    return (vec_iout,sig)

def calc_sign(isite,vec_i,p,nc):
#    print "vec_isite",vec_i[isite],p
    if vec_i[isite] == p:
        sig = 0
    else:
        sig = 1
        isum = np.sum(vec_i[isite+1:2*nc])
        sig = (-1)**(isum)
        
    return sig

def calc_ii2vec(ii,nc):
    vec_i = np.arange(nc*2)
    iii = ii
    vec_i[0]=iii%2
#    print vec_i[0]
#    print vec_i[0]
    iii = (ii-vec_i[0])/2
#    print ii
    for i in range(2*nc-1):        
        vec_i[i+1] = iii%2
#        print vec_i[i+1]
        iii = (iii-vec_i[i+1])/2

#    print vec_i
    return vec_i

def exact_init(nx,ny):
    nc = nx*ny
    nf = 4**(nc)
    mat_cvec = ()
    mat_cdvec = ()
    mat_cdc = lil_matrix((nf,nf))

    for isite in range(nc*2):
        mat_c = calc_matc(isite,nx,ny,0)
#        print isite,mat_c
        mat_cvec += (mat_c,)
        mat_cdc = mat_c.T
        mat_cdvec +=(mat_cdc,)

        
    return (mat_cvec,mat_cdvec)


def exact_init_fix(nx,ny,nup,ndown,mf):
    nc = nx*ny
    nf = 4**nc
    mat_cvec = ()
    mat_cdvec = ()


    
    vec_hi = calc_map(nc,nf,nup,ndown,mf)

    mup = scm.comb(nc,nup-1,1)
    mdown = scm.comb(nc,ndown,1)
    mf2 = mup*mdown
    
    vec_hi2 = calc_map(nc,nf,nup-1,ndown,mf2)

    mup = scm.comb(nc,nup,1)
    mdown = scm.comb(nc,ndown-1,1)
    mf3 = mup*mdown
    
    vec_hi3 = calc_map(nc,nf,nup,ndown-1,mf3)

    for ispin in range(2):
        for isi in range(nc):
            isite = ispin*nc + isi
            if ispin == 0:                
                mat_c = calc_matc_fix(isite,nx,ny,nup,ndown,mf,mf2,vec_hi,vec_hi2,0)
            else :
                mat_c = calc_matc_fix(isite,nx,ny,nup,ndown,mf,mf3,vec_hi,vec_hi3,0)
                
            mat_cvec += (mat_c,)
            mat_cdc = mat_c.T
            mat_cdvec +=(mat_cdc,)




    return (mat_cvec,mat_cdvec)


def const_h(nx,ny,nn,mu,U,mat_cvec,mat_cdvec):
    mat_h = lil_matrix((nn,nn))
    mat_temp = lil_matrix((nn,nn))
    mat_temp2 = lil_matrix((nn,nn))
    
    
    for ix in range(nx):
        for iy in range(ny):
            for ispin in range(2):
                isite = (ispin)*nx*ny+iy*nx+ix
                jspin = ispin
                jx = ix + 1
                jy = iy
                if jx >= nx and nx != 1:
                    jx = jx -nx

                jsite = jspin*nx*ny+jy*nx+jx
                if jx < nx:
                    v = -1.0
                    mat_c = mat_cvec[jsite]
                    mat_cdc = mat_cdvec[isite]
                    mat_h += v*mat_cdc*mat_c

                jx = ix - 1
                jy = iy

                if jx < 0 and nx !=1:
                    jx = jx +nx

                
                jsite = jspin*nx*ny+jy*nx+jx
                if jx >= 0:
                    v = -1.0
                    mat_c = mat_cvec[jsite]
                    mat_cdc = mat_cdvec[isite]
                    mat_h += v*mat_cdc*mat_c

                jx = ix
                jy = iy+1
                if jy >= ny and ny != 1:
                    jy = jy -ny


                jsite = jspin*nx*ny+jy*nx+jx
                if jy < ny:
                    v = -1.0
                    mat_c = mat_cvec[jsite]
                    mat_cdc = mat_cdvec[isite]
                    mat_h += v*mat_cdc*mat_c

                jx = ix
                jy = iy -1
                if jy < 0 and ny != 1:
                    jy = jy +ny


                jsite = jspin*nx*ny+jy*nx+jx
                if jy >= 0:
                    v = -1.0
                    mat_c = mat_cvec[jsite]
                    mat_cdc = mat_cdvec[isite]
                    mat_h += v*mat_cdc*mat_c

                jx = ix
                jy = iy

                jsite = jspin*nx*ny+jy*nx+jx

                v = -mu
                mat_c = mat_cvec[jsite]
                mat_cdc = mat_cdvec[isite]
                mat_h += v*mat_cdc*mat_c

    for ix in range(nx):
        for iy in range(ny):
            ispin = 1
            isite = ispin*nx*ny+iy*nx+ix
            mat_c = mat_cvec[isite]
            mat_cdc = mat_cdvec[isite]
            mat_temp = mat_cdc*mat_c
#            print "c",mat_c
#            print "cd",mat_cdc
#            print "1",mat_temp
            ispin = 0
            isite = ispin*nx*ny+iy*nx+ix
            mat_c = mat_cvec[isite]            
            mat_cdc = mat_cdvec[isite]
            mat_temp2 = mat_cdc*mat_c
#            print "2",mat_temp2
#            print "3",mat_temp2*mat_temp
            mat_h += U*mat_temp2*mat_temp
            

    mat_h=mat_h.tocsr()
#    print "Hamitonian"
#    print mat_h
                
        
    return mat_h


def calc_map(nc,nf,nup,ndown,mf):    
    vec_hi = np.full((nf), -1, dtype=int)

    ii = 0
    for i in range(nf):
        vec_i = calc_ii2vec(i,nc)
        num_up = np.sum(vec_i[0:nc])
        num_down = np.sum(vec_i[nc:2*nc])

        if num_up == nup and num_down == ndown:
            vec_hi[i] = ii
            ii += 1
#    print vec_hi
    return vec_hi


def main():
    print "(^_^)"
    U = -2.0
    mu = U/2
    nx = 2
    ny = 2
    beta = 10.0
    nc = nx*ny
    nf = 4**nc
    print "Nx x Ny:",nx,ny
#    print "Temperature:",1.0/beta
    print "U:",U
    print "mu:",mu


#------------------------------------------------
    

#    fulldiag = True
    fulldiag = False    
    
    if fulldiag:
        print "----------------------------------------"
        print "Dimension:",nf

#        mat_h = exact_init(nx,ny,mu,U)

        mat_cvec,mat_cdvec=exact_init(nx,ny)
        mat_h = const_h(nx,ny,nf,mu,U,mat_cvec,mat_cdvec)

        x = sc.rand(nf,1)
        
        w,v = sc.sparse.linalg.lobpcg(mat_h,x,largest=None)
        print "Minimum eigenvalue",w
        print "----------------------------------------"

    

#Number conservation with each spin is used-----
#    nfix = True
    nfix = False

    nup = nc/2
    ndown = nc/2
    if nfix:        
        print "----------------------------------------"
        print "Numbers of each spin are fixed"
        print "Num. of up spins:",nup
        print "Num. of down spins",ndown

        mup = scm.comb(nc,nup,1)
        mdown = scm.comb(nc,ndown,1)
        mf = mup*mdown
        print  "Dimension with fixed n:",mf

        mat_cvecf,mat_cdvecf=exact_init_fix(nx,ny,nup,ndown,mf)
        mat_hr = const_h(nx,ny,mf,mu,U,mat_cvecf,mat_cdvecf)
        
        xf = sc.rand(mf,1)
        wf,vf = sc.sparse.linalg.lobpcg(mat_hr,xf,largest=None)
        print "Minimum eigenvalue with fixed n:",wf
        print "----------------------------------------"

#------------------------------------------------





if __name__ == "__main__":
    main()
    


