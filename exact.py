from scipy.sparse import lil_matrix,csr_matrix
#from scipy.linalg
from scipy.sparse import linalg
from numpy.linalg import norm
import numpy as np
import scipy as sc

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

def exact_init(nx,ny,mu,U):
    nc = nx*ny
    nf = 4**(nc)
    mat_cvec = ()
    mat_cdvec = ()
    mat_cdc = lil_matrix((nf,nf))
    mat_temp = lil_matrix((nf,nf))
    mat_temp2 = lil_matrix((nf,nf))
    mat_h = lil_matrix((nf,nf))

    for isite in range(nc*2):
        mat_c = calc_matc(isite,nx,ny,0)
#        print isite,mat_c
        mat_cvec += (mat_c,)
        mat_cdc = mat_c.T
        mat_cdvec +=(mat_cdc,)


    for ix in range(nx):
        for iy in range(ny):
            for ispin in range(2):
                isite = (ispin)*nx*ny+iy*nx+ix
                jspin = ispin
                jx = ix + 1
                jy = iy
                if jx >= nx:
                    jx = jx -nx

                jsite = jspin*nx*ny+jy*nx+jx
                if jx < nx:
                    v = -1.0
                    mat_c = mat_cvec[jsite]
                    mat_cdc = mat_cdvec[isite]
                    mat_h += v*mat_cdc*mat_c

                jx = ix - 1
                jy = iy

                if jx < 0:
                    jx = jx +nx

                
                jsite = jspin*nx*ny+jy*nx+jx
                if jx >= 0:
                    v = -1.0
                    mat_c = mat_cvec[jsite]
                    mat_cdc = mat_cdvec[isite]
                    mat_h += v*mat_cdc*mat_c

                jx = ix
                jy = iy+1
                if jy >= ny:
                    jy = jy -nx


                jsite = jspin*nx*ny+jy*nx+jx
                if jy < ny:
                    v = -1.0
                    mat_c = mat_cvec[jsite]
                    mat_cdc = mat_cdvec[isite]
                    mat_h += v*mat_cdc*mat_c

                jx = ix
                jy = iy -1
                if jy < 0:
                    jy = jy +nx


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


def main():
    print "(^_^)"
    U = -2.0
    mu = U/2-1.0
    nx = 2
    ny = 2
    beta = 10.0
    nc = nx*ny
    nf = 4**nc
    print "Nx x Ny:",nx,ny
#    print "Temperature:",1.0/beta
    print "Dimension:",nf
    print "U:",U
    print "mu:",mu

    mat_h=exact_init(nx,ny,mu,U)
    x = sc.rand(nf,1)
    
    w,v = sc.sparse.linalg.lobpcg(mat_h,x,largest=None)
    print "Minimum eigenvalue",w
    


if __name__ == "__main__":
    main()
    
