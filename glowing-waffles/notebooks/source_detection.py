from ccdproc import CCDData
from astropy.stats import sigma_clipped_stats
from photutils import daofind

ccd = CCDData.read('/Users/Nathan/Desktop/Astro-Project/Testing/M52-002R.fit', unit="adu")

def source_detection(ccd):
	data = ccd.data    
	mean, median, std = sigma_clipped_stats(data, sigma=3.0, iters=5)    
	sources = daofind(data - median,fwhm=3.0, threshold=5.*std)
	return sources