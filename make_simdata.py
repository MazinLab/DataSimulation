import mkidsim.utils as utils
import scipy.ndimage.filters as filter
from mkidpipeline.utils import plottingTools as pt
import numpy as np
import mkidpipeline.hotpix.generatebadpixmask as gbpm
import random


filename='/mnt/data0/isabel/sandbox/CHARIS/images.fits'
images,hdr= utils.open_file(filename)

testimage=np.copy(images[0])
random_xpix = [random.randrange(50, 135, 1) for _ in range(50)]
random_ypix = [random.randrange(50, 135, 1) for _ in range(50)]
random_values = [random.randrange(3000, 5000, 100) for _ in range(50)]

for i in np.arange(len(random_values)): testimage[random_xpix[i],random_ypix[i]]=random_values[i]
pt.plot_array(testimage)

#gbpm.check_interval(testimage, fwhm=2.5, box_size=5, nsigma_hot=3.0, max_iter=5,
#                   use_local_stdev=False, bkgd_percentile=50.0)

gbpm.hpm_laplacian(testimage, box_size=5, nsigma_hot=4.0)
