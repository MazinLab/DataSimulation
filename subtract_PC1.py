import mkidsim.utils as utils
from mkidpipeline.utils import plottingTools as pt
from mkidpipeline.speckle.genphotonlist_IcIsIr import genphotonlist2D as gen2D
import numpy as np
import scipy.optimize as opt


def handpickedlist_calculate_scalefactor(meanimage, basis_1):
    seeing_disk_points=[[87,111],[109,102],[79,96],[101,76],[85,117]]
    scale_factor_Ic_list=[]
    for i in np.arange(len(seeing_disk_points)):
        point=seeing_disk_points[i]
        scale_factor_Ic=-meanimage[point[1],point[0]]/basis_1[point[1],point[0]] #image was flipped when I plotted it
        scale_factor_Ic_list.append(scale_factor_Ic)
    median_scale_factor_Ic = np.median(scale_factor_Ic_list)

    diffrac_im_points=[[113,98],[77,88],[92,102],[70,84],[87,104]]
    scale_factor_diffrac_list=[]
    for i in np.arange(len(diffrac_im_points)):
        point=diffrac_im_points[i]
        scale_factor_diffrac=meanimage[point[0],point[1]]/basis_1[point[0],point[1]]
        scale_factor_diffrac_list.append(scale_factor_diffrac)
    median_scale_factor_diffrac=np.median(scale_factor_diffrac_list)

    Ic_image=meanimage+median_scale_factor_Ic*basis_1
    Seeing_image=meanimage-median_scale_factor_Ic*basis_1

    return(Ic_image, Seeing_image)

def seeing_radius_calculate_scalefactor(meanimage, basis_1, n=181, xcenter=94, ycenter=94, r1=12, r2=18, calc_method='l1_norm'):
    y,x=np.ogrid[-xcenter:n-xcenter, -ycenter:n-ycenter]
    mask_array=np.zeros((n,n))
    mask_inner=x*x+y*y <=r1*r1
    mask_outer=x*x+y*y <=r2*r2
    mask_array[mask_outer]=1
    mask_array[mask_inner]=0

    im_tofit_mean=meanimage*mask_array
    im_tofit_basis=basis_1*mask_array
    basis_factors=im_tofit_basis[np.where(im_tofit_basis < 0)]
    mean_factors=im_tofit_mean[np.where(im_tofit_basis < 0)]

    if calc_method=='median':
        median_scale_factor_Ic=np.median(-mean_factors/basis_factors)
        Ic_image=meanimage+median_scale_factor_Ic*basis_1
        Seeing_image=meanimage-median_scale_factor_Ic*basis_1

    if calc_method=='l2_norm':
        basis_tofit = []
        for i in np.arange(len(basis_factors)): basis_tofit.append([basis_factors[i], 0])
        l2fit_scale_factor_Ic = -np.linalg.lstsq(basis_tofit, mean_factors, rcond=None)[0][0]
        Ic_image=meanimage+l2fit_scale_factor_Ic*basis_1
        Seeing_image=meanimage-l2fit_scale_factor_Ic*basis_1

    if calc_method=='l1_norm':
        basis_tofit = []
        for i in np.arange(len(basis_factors)): basis_tofit.append([basis_factors[i], 0])
        initial_guess = -np.linalg.lstsq(basis_tofit, mean_factors, rcond=None)[0][0]

        def fun_tofit(x):
            return x*basis_factors+mean_factors

        l1fit_scale_factor_Ic=opt.least_squares(fun_tofit, initial_guess, loss='soft_l1').x[0]
        Ic_image=meanimage+l1fit_scale_factor_Ic*basis_1
        Seeing_image=meanimage-l1fit_scale_factor_Ic*basis_1

    return(Ic_image, Seeing_image, im_tofit_basis, im_tofit_mean)

def generate_Ic_Is(images, Ic_image, Seeing_image):
    Im=np.reshape(images[0], 181*181)
    Bad=np.reshape(Seeing_image, 181*181)
    Perfect=np.reshape(Ic_image, 181*181)
    To_Fit=[]
    for i in np.arange(181*181): To_Fit.append([Perfect[i], Bad[i]])

    Im_solution=np.linalg.lstsq(To_Fit, Im, rcond=None)[0]
    Ic=Ic_image*Im_solution[0]
    Is=images[0]-Ic

    return(Ic, Is)

if __name__ == "__main__":
    images, meanimage, basis_1, hdr, D, U, W, VT = utils.PCA_2d('/mnt/data0/isabel/sandbox/CHARIS/images.fits', 13,'/mnt/data0/isabel/sandbox/CHARIS/13components.fits')
    Ic_image, Seeing_image, im_tofit_basis, im_tofit_mean = seeing_radius_calculate_scalefactor(meanimage, basis_1, calc_method='l1_norm')
    Ic, Is = generate_Ic_Is(images, Ic_image, Seeing_image)
#    gen2D(Ic, Is, 0, 5, 0.1, '/mnt/data0/isabel/sandbox/CHARIS/testnewcode.h5', deadtime=10, interpmethod='cubic', taufac=500, diffrac_lim=2.86)
    np.save('/mnt/data0/isabel/sandbox/CHARIS/Ic', Ic)
    np.save('/mnt/data0/isabel/sandbox/CHARIS/Is', Is)