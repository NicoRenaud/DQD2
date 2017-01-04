import sys
import numpy as np
import warnings 
import subprocess as sp
tmp = sp.call('cls',shell=True)

print " "
print " ==============================================="
print " ==  SKIMAGE QD Detector                      =="
print ' ==  Nicolas Renaud TU Delft 2016             =='
print " ==============================================="
print " "

print " == Import matplotlib module"
import matplotlib.pyplot as plt

# skimage modules
print " == Import skimage modules"
import skimage.io
from skimage.util import img_as_ubyte,img_as_uint,img_as_int
from skimage.restoration import denoise_tv_chambolle 
from skimage.filters.rank import otsu, median
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.morphology import disk
from skimage.feature import  canny
from skimage.measure import label as sk_label
from skimage import exposure

#scipy modules
print " == Import scipy modules"
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label, find_objects,generate_binary_structure

# other modules
import argparse
from tqdm import *


# flags
_plot_  = 1

# there is a nasty warning that bugs me
# when changing image from float to int
warnings.filterwarnings('ignore',category=UserWarning, append=True)

def main(argv):


    ##########################
    ##  Import the arguments
    ##########################

    parser = argparse.ArgumentParser(description='Identify dots in SEM pictures')
    
    # image to process
    parser.add_argument('image_name', help="File containing the picture",type=str)

    # size of the image (182.62 nm for A xxx for B)
    parser.add_argument('size_image', help="Dimension in nm of the image",type=float)
   
        
    #optional arguments

    # initial denoising of the image
    parser.add_argument('-ud', '--use_denoise', default=0,
                        help = ' Denoise the picture before treating it (0/1)',type=int)

    parser.add_argument('-dw', '--denoise_weight', default=2.0,
                        help = ' Denoise weight (default 2.0)',type=float)

    parser.add_argument('-upf', '--use_prefiltering', default=1,
                        help = ' Use a Gaussin filter to denoise the original image (0/1)',type=int)

    parser.add_argument('-pfw', '--prefiltering_weight', default=20.0,
                        help = ' prefilter Gaussian weight (default 7.0)',type=float)





    # enhance contrast
    parser.add_argument('-uhe', '--use_hist_equal', default=0,
                    help = ' Use an histogram equalization to enhance contrast (0/1)',type=int)




    # thresholding : Global Otsu or Local Otsu or Local (Adatative)
    parser.add_argument('-use_global_otsu', '--use_global_otsu', default=0,
                        help = ' Use Otsu global thresholding',type=int)

    parser.add_argument('-use_local_otsu', '--use_local_otsu', default=0,
                        help = ' Use Otsu local thresholding',type=int)

    parser.add_argument('-otsu', '--otsu_parameter', default=10,
                        help = ' Parameter for the Otsu thresholding',type=int)

    parser.add_argument('-use_adapt_th', '--use_adatative_threshold', default=1,
                        help = ' Use adaptative thresholding',type=int)

    parser.add_argument('-ad_block', '--adaptative_block', default=300,
                        help = ' block size for adatative thresholding',type=int)

    parser.add_argument('-ad_offset', '--adaptative_offset', default=5,
                        help = ' offset for adatative thresholding',type=int)




    # post filtering
    parser.add_argument('-gf', '--Gaussian_fitlering', default=0.,
                        help = ' Standard devitation for the Gaussian flitering',type=float)




    # canny edge detection
    parser.add_argument('-sc', '--sigma_canny', default=1.0,
                        help = ' Standard devitation for the Canny edge detection',type=float)


    # max size for the dots
    parser.add_argument('-dmax', '--diameter_max', default=25.0,
                        help = ' Maximum diameter of the dots (default 10 nm)',type=float)

    parser.add_argument('-dmin', '--diameter_min', default=0.0,
                        help = ' Minimum diameter of the dots (default 0.5 nm)',type=float)


    # min/max  apsect ration
    parser.add_argument('-ar_min' , default=0.0,
                        help = ' Minimum aspect ratio for the dots',type=float)

    args=parser.parse_args()

    
    
    ##########################   
    ## Load picture 
    ##########################   

    print '\n == Loading Image : %s' %args.image_name
    image_ori = skimage.io.imread(args.image_name, as_grey = True)
    nPixel = len(image_ori)
    print ' == Image size : %dx%d pixel, %1.2fx%1.2f nm' %(nPixel,nPixel,args.size_image,args.size_image)


    #################################   
    ## denoise the picture 
    #################################

    # use a tv-chambolle denoising
    # slow but efficient
    if args.use_denoise:
        print ' == Denoise Image (Chambolle): weight = %1.3f' %args.denoise_weight
        image = denoise_tv_chambolle(image_ori, weight=args.denoise_weight, multichannel=True)

    # use a simple Gaussian filtering
    # fast but not super efficient
    elif args.use_prefiltering:
        print ' == Denoise Image (Gaussian): weight = %1.3f' %args.prefiltering_weight
        image = gaussian_filter(image_ori, args.prefiltering_weight)

    # do not prefilter the image
    else:
        print ' == Warning : No Denoising of the Image' 
        image = image_ori

    #np.savetxt('denoised.dat',image)
    #image = np.loadtxt('denoised.dat')
    
    #################################   
    ## enhabce contrast with 
    #################################
    if args.use_hist_equal:
        print '\n == Enhance Contrast'
        image = exposure.equalize_hist(image)
    else:
        image = image
    

    #################################
    ## global Otsu Thresholding, 
    ## t_glob_otsu is a scalar
    #################################
    if args.use_global_otsu:

        print ' == Global Otsu Threshold : paramter = %1.3f' %args.otsu_parameter
        
        p8 = image
        p8 = median(p8, disk(args.otsu_parameter))
        t_glob_otsu = threshold_otsu(p8)
        image_thr = p8 >= t_glob_otsu
        image_thr = img_as_int(image_thr)

    elif args.use_local_otsu:

        print ' == Local Otsu Threshold : paramter = %1.3f' %args.otsu_parameter
        p8 = image
        t_loc_otsu = otsu(p8, disk(args.otsu_parameter))
        image_thr = p8 <= t_loc_otsu
        image_thr = img_as_int(image_thr)

    elif args.use_adatative_threshold:

        print ' == Adatative Threshold : block = %d offset %d' %(args.adaptative_block,args.adaptative_offset)
        image_thr = threshold_adaptive(image, args.adaptative_block, offset=args.adaptative_offset)

    else:
        print ' == Error no Thresholding required '
        sys.exit()

    #np.savetxt('otsu.dat',glob_otsu)
    #glob_otsu_or = np.loadtxt('otsu.dat')
    



    #################################
    ## Gaussian Filtering 
    #################################

    print ' == Gaussian filtering : weight = %1.3f' %args.Gaussian_fitlering
    image_thr_gf = gaussian_filter(image_thr, args.Gaussian_fitlering)
    
    #################################
    ## Canny edge detection
    #################################

    print ' == Canny edges detection: sigma = %1.3f' %args.sigma_canny 
    edges = canny(image_thr_gf, sigma=args.sigma_canny)
    edges = img_as_uint(edges)


    #################################
    ## Find object using scipy
    #################################

    print ' == Find objects'
    #s = generate_binary_structure(2,2)
    #labeled_array, num_features = label(edges,structure=s)
    labeled_array, num_features = sk_label(edges,connectivity=2,return_num=True)

    #################################
    ## measure objects
    #################################

    print ' == Analyse %d detected objects' %num_features
    print '    Keep dots between  %1.2f - %1.2f nm in diameter' %(args.diameter_min,args.diameter_max)
    print '    Keep dots with AR larger than %1.2f ' %(args.ar_min)

    cont, center, size,aspect_ratio = [], [], [], []
    
    # measure the size for each object
    for i in tqdm(range(num_features),leave=False,ncols=40):

        # find where the object is
        ind = np.argwhere(labeled_array==i+1)
        
        # compute the center of the object
        if len(ind)>0:

            c = [np.mean(ind[:,0]),np.mean(ind[:,1])]
            
            # compute the distance of each point to the center
            radius = []
            nPixel = len(image_ori)
            scale = args.size_image/nPixel

            for l in range(len(ind)):
                r = np.sqrt(np.sum((ind[l]-c)**2))*scale
                radius.append(r)

            # compute mean, min and max distance
            rmin = np.min(radius)
            rmax = np.max(radius)
            rmean = np.mean(radius)
            

            # store that if max > 0 and smaller than max size
            if (rmax>0) and (2*rmean<args.diameter_max) and (2*rmean>args.diameter_min) :

                # store only if the aspect ration is ok
                ar = float(rmin)/rmax
                if  ar > args.ar_min :
                    size.append(2*rmean)
                    aspect_ratio.append(ar)
                    cont.append(ind)
                    center.append(c)

    # need that for plotting
    center=np.array(center)


    ######################################
    ## plot the results of the analysis
    ######################################

    if _plot_:

        print ' == Plot the image analysis summary'
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(30, 20), sharex=True, sharey=True)
        fig.subplots_adjust(bottom=0.05, left=0.05, right=0.95, top=0.95, wspace=0., hspace=0.)
        
        ax1.imshow(image_ori, cmap=plt.cm.gray, interpolation='nearest')
        ax1.axis('off')
        ax1.set_title('original')

        
        ax2.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
        ax2.axis('off')
        ax2.set_title('denoised')

        
        ax3.imshow(image_thr, cmap=plt.cm.gray, interpolation='nearest')
        ax3.axis('off')
        ax3.set_title('threshold')

        
        ax4.imshow(image_thr_gf, cmap=plt.cm.gray, interpolation='nearest')
        ax4.axis('off')
        ax4.set_title('filter')

        ax5.imshow(edges,cmap=plt.cm.gray_r,interpolation=None)
        ax5.axis('off')
        ax5.set_title('edges')

        ax6.imshow(image_ori,cmap=plt.cm.gray)
        num_features_kept = len(cont)
        for i in tqdm(range(num_features_kept),leave=False,ncols=40):
            ax6.plot(cont[i][:,1],cont[i][:,0],'-o',color='#004FFF',markersize=1.,alpha=0.5)
            ax6.plot(center[i,1],center[i,0],'o',color='black',markersize=2.,alpha=0.5)
        ax6.axis('off')
        ax6.set_title('detected')
        
        print '    save image'
        plt.savefig('bilan.jpg')
        plt.close()

    ######################################
    ## plot the size/aspect ratio of
    ## the detected objects
    ######################################

    if _plot_ :

        print ' == Plot the size distribution'
        fig, (ax1,ax2) = plt.subplots(1, 2, sharex=False, sharey=False)

        
        ax1.hist(size,bins=50)
        ax1.set_xlabel('size (nm)')
        ax1.set_xlim([0,10])


        ax2.hist(aspect_ratio,bins=50)
        ax2.set_xlabel('Aspect ratio')
        ax2.set_xlim([0,1])        

        plt.savefig('size.jpg')
        plt.close()

    ######################################
    ## plot the deteted objects only
    ######################################

    if _plot_ :

        plt.figure(num=None, figsize=(12, 12), dpi=100)
        print ' == Plot the detected objects'
        plt.imshow(image_ori,cmap=plt.cm.gray)
        for i in tqdm(range(num_features_kept),leave=False,ncols=40):
            plt.plot(cont[i][:,1],cont[i][:,0],'-o',color='#004FFF',markersize=1.,alpha=0.5)
            plt.plot(center[i,1],center[i,0],'o',color='black',markersize=2.,alpha=0.5)
        plt.axis('off')
        print '    save image'
        plt.savefig('detect.jpg')

if __name__=='__main__':
    main(sys.argv[1:])
