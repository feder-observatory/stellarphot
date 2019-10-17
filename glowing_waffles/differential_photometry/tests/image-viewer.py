import numpy as np

from astropy.visualization import scale_image
from astropy.nddata import block_replicate
from astropy.table import Table, Column
import astropy.units as u

import astropy.coordinates as apycoord

from catalog_search import catalog_search
from filter_transform_ivezic_2007 import filter_transform

from ccdproc import CCDData


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# # Read image

# In[2]:

ccd = CCDData.read('kelt-1-b-054R.fit', unit='adu')


# In[3]:

# This function really should be somewhere else eventually.
def scale_and_downsample(data, downsample=4,
                         min_percent=20,
                         max_percent=99.5):

    scaled_data = scale_image(data,
                              min_percent=min_percent,
                              max_percent=max_percent)

    if downsample > 1:
        scaled_data = block_reduce(scaled_data,
                                   block_size=(downsample, downsample))
    return scaled_data


def uniformize_source_names(aij_tbl):
    import re

    data_col = re.compile(r'Source-Sky_[TC](\d+)')
    sources = []
    for c in aij_tbl.colnames:
        match = re.search(data_col, c)
        if match:
            sources.append(len(sources) + 1)
            source_number = match.groups()[0]
            try:
                aij_tbl.rename_column(c, 'Source-Sky_C' + source_number)
            except KeyError:
                # Column already exists s there was no need to change the name
                pass
            try:
                aij_tbl.rename_column('Source_Error_T' + source_number,
                                      'Source_Error_C' + source_number)
            except KeyError:
                pass
            try:
                aij_tbl.rename_column('Peak_T' + source_number,
                                      'Peak_C' + source_number)
            except KeyError:
                pass
            try:
                aij_tbl.rename_column('RA_T' + source_number,
                                      'RA_C' + source_number)
            except KeyError:
                pass
            try:
                aij_tbl.rename_column('DEC_T' + source_number,
                                      'DEC_C' + source_number)
            except KeyError:
                pass
            try:
                aij_tbl.rename_column('X(FITS)_T' + source_number,
                                      'X(FITS)_C' + source_number)
            except KeyError:
                pass
            try:
                aij_tbl.rename_column('Y(FITS)_T' + source_number,
                                      'Y(FITS)_C' + source_number)
            except KeyError:
                pass

    return sources


def source_column(source_number):
    col_name = 'Source-Sky_C' + str(source_number)
    return col_name


def source_ra(source_number):
    col_name = 'RA_C' + str(source_number)
    return col_name


def source_dec(source_number):
    col_name = 'DEC_C' + str(source_number)
    return col_name


def source_x_pix(source_number):
    col_name = 'X(FITS)_C' + str(source)
    return col_name


def source_y_pix(source_number):
    col_name = 'Y(FITS)_C' + str(source)
    return col_name


def source_error(source_number):
    col_name = 'Source_Error_C' + str(source_number)
    return col_name


# In[4]:

apass, apass_x, apass_y = catalog_search(
    ccd.wcs, ccd.shape, 'II/336/apass9', 'RAJ2000', 'DEJ2000')
apass_bright = (apass['e_r_mag'] < 0.05) & (
    apass['u_e_r_mag'] == 0) & (apass['e_B-V'] < 0.1)
apass_in_bright, in_apass_x, in_apass_y = apass[apass_bright], apass_x[apass_bright], apass_y[apass_bright]

vsx, vsx_x, vsx_y = catalog_search(
    ccd.wcs, ccd.shape, 'B/vsx/vsx', 'RAJ2000', 'DEJ2000')
vsx_names = vsx['Name']
print(ccd.shape)
print(type(ccd))


# In[5]:

disp = scale_and_downsample(ccd.data, downsample=1)


# In[6]:

import mpld3
mpld3.enable_notebook()
# mpld3.disable_notebook()


# In[7]:

plt.figure(figsize=(12, 7))
plt.imshow(disp, cmap='gray', origin='lower')
plt.scatter(vsx_x, vsx_y, c='none', s=100, edgecolor='cyan')
plt.title('Blue: VSX, Yellow: APASS', fontsize=20)

for x, y, m in zip(vsx_x, vsx_y, vsx_names):
    plt.text(x, y, str(m), fontsize=18, color='cyan')

plt.scatter(in_apass_x, in_apass_y, c='none', s=50,
            edgecolor='yellow', alpha=0.5, marker='o')
# for x, y, c in zip(apass_x, apass_y, apass_in):
#    plt.text(x, y, c.to_string(), fontsize=12, color='yellow')

plt.xlim(0, ccd.shape[1])
plt.ylim(0, ccd.shape[0])


# In[8]:

apass_mags = apass_in_bright['r_mag']
apass_ra = apass_in_bright['RAJ2000']
apass_dec = apass_in_bright['DEJ2000']
apass_color = apass_in_bright['B-V']

aij_raw = Table.read('Measurements_2013-09-13_R_clipping.csv')
image_index = [el for el in range(len(aij_raw['dumbname']))]
sources = uniformize_source_names(aij_raw)
source_index = [el for el in range(len(sources))]
aij_ra = []
aij_dec = []
aij_mags = []
for source in sources:
    aij_ra.append(aij_raw[source_ra(source)][len(aij_raw['dumbname']) // 2])
    aij_dec.append(aij_raw[source_dec(source)][len(aij_raw['dumbname']) // 2])
    aij_mags.append(-2.5 * np.log10(aij_raw[source_column(source)]))
aij_mags = np.array(aij_mags)
aij_coordinates = apycoord.SkyCoord(aij_ra, aij_dec, unit=(u.hour, u.deg))
apass_coordinates = apycoord.SkyCoord(apass_ra, apass_dec, unit='deg')


# In[9]:

match = apycoord.match_coordinates_sky(aij_coordinates, apass_coordinates)


# In[ ]:


# #APASS Filter Corrections
# ##Transform the APASS r magnitudes into R magnitudes using APASS r and i magnitudes
#
# The equation used is R-feder - r-apass = A*c**3 + B*c**2 + C*c + D
#
# Where...

# In[10]:

A = -0.0107
B = 0.0050
C = -0.2689
D = -0.1540


# And the value c is dependant on each object and is equal to the i magnitude subtracted from the r magnitude.
#
# c = r-i
#
# The initial equation can be solved for BVRIfeder by simply adding griapass on both sides.
#
# R-feder = Ac**3 + Bc*2 + Cc + D + r-apass

# In[11]:

apass_r_mags = apass_in_bright['r_mag']
apass_i_mags = apass_in_bright['i_mag']
c = apass_r_mags - apass_i_mags
apass_R_mags = (A * (c**3)) + (B * (c**2)) + (C * c) + D + apass_r_mags


# In[ ]:


# In[12]:

transformed = filter_transform(apass_in_bright, 'R', r='r_mag', i='i_mag')
transformed.description


# In[13]:

print(transformed)


# In[14]:

get_ipython().magic(u'matplotlib inline')
corrections = []
fit_error = []
for image in image_index:
    BminusV = []
    Rminusr = []
    for index, el in enumerate(match[0]):
        if aij_coordinates[index].separation(apass_coordinates[el]).arcsec < 0.5:
            if aij_mags[index][image] < 100:
                BminusV.append(apass_color[el])
                Rminusr.append(apass_R_mags[el] - aij_mags[index][image])
    slope_intercept, cov = np.polyfit(BminusV, Rminusr, 1, cov=True)
    corrections.append(slope_intercept)
    """The Error in the slope and the intercept are the diagonals of the matrix cov.
    The first diagonal ([0,0]) is the error in the slope and the second diagonal
    is the error in the intercept. See the Documentation for polyfit for more info"""
    fit_error.append([np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])])


# In[16]:


good_pics = image_index
slope_error = []
intercept_error = []
for el in fit_error:
    slope_error.append(el[0])
    intercept_error.append(el[1])


# In[17]:

kelt_1_data = []
keltBminusV = 0.6  # MUST HAVE B-V FOR ALL STARS IN FIELD!!!!#####################
corrected_curves = np.zeros((len(sources), len(good_pics)))
corrected_curves_er = np.zeros((len(sources), len(good_pics)))
all_SNR = np.zeros((len(good_pics), len(sources)))
for obj in source_index:
    tmp_data = []
    tmp_data_er = []
    plt.figure()
    for image in good_pics:
        # Date
        JD = aij_raw['J.D.-2400000'][image]
        # Magnitude
        color_term = keltBminusV * corrections[image][0]
        intercept = corrections[image][1]
        old_mag = aij_mags[obj][image]
        new_mag = old_mag + intercept + color_term
        tmp_data.append(new_mag)
        # Error
        # Error in color term
        #Error in slope
        frac_unc_slope = slope_error[image] / corrections[image][0]
        #Error in B-V
        er_BV = 0.005  # Estimation
        frac_unc_BV = er_BV / keltBminusV

        color_err = np.sqrt(frac_unc_slope**2 + frac_unc_BV**2) * color_term
        # Error in intercept term
        er_intercept = intercept_error[image]
        # Error in old mag
        SNR = aij_raw[source_column(obj + 1)][image] / \
            aij_raw[source_error(obj + 1)][image]
        er_old_mag = 1 / SNR
        all_SNR[image, obj] = 1.0 * SNR

        er = np.sqrt(er_old_mag**2 + er_intercept**2 + color_err**2)
        tmp_data_er.append(er)
        # Plot
        plt.errorbar(JD, new_mag, yerr=er, color='grey')
        plt.errorbar(JD, new_mag, color='red', xerr=0.0005)
    corrected_curves[obj] = np.array(tmp_data)
    corrected_curves_er[obj] = tmp_data_er
    # plt.title('Source_C'+str(obj+1))
    # plt.show()


# ###Check what the slope intercepts and slope values are over the course of the night

# In[34]:

# Check the slope intercept over the night
for index, slopeinfo in enumerate(corrections):
    plt.scatter(index, slopeinfo[1])
plt.title('intercept')
plt.show()

# Check the slope values over the night
for index, slopeinfo in enumerate(corrections):
    plt.scatter(index, slopeinfo[0])
plt.title('slope')
plt.show()


# #Finding Comparison Stars
# ##An automated way of finding the best comparison stars from the apass calibrated data
#
# For each source, the standard deviation is calculated. This deviation is the mulitplied by the sources average magnitude. The sources with the lowest values have an optimal combination of being fairly stable and fairly bright.

# #Comparison Set Calculations
# ## With Error

# In[76]:

def calc_comp_weights(images, comparison_sources, SNR_data):
    # Calculate the initial weights for each source for each image by deviding the sources SNR^2 by the norm const for that image
    """norm_consts = np.zeros(len(images))
    for image in images:
        #set the sum equal to zero
        SNR2sum = 0
        for source in comparison_sources:
            #add the SNR^2 to the sum
            SNR2sum += SNR_data[image-1,source-1]**2
        #set the norm constant for that image equal to one over the SNR^2 sum
        norm_consts[image-1] = 1/SNR2sum

    weights = np.zeros((len(images),len(comparison_sources)))
    for image, norm in zip(images, norm_consts):
        for source in comparison_sources:
            weights[image-1,source-1] = norm*(SNR_data[image-1,source-1]**2)"""

    comparison_sources = np.array(comparison_sources)
    norm_const = 1 / np.sum((SNR_data[0, :]**2))

    weights = np.zeros(len(comparison_sources))
    for source in comparison_sources:
        weights[source - 1] = norm_const * (SNR_data[0, source - 1]**2)
    return weights


def calc_comp_star(images, comparison_sources, mag_data, weights):
    vert_comp_star = np.zeros(len(images))
    for image in images:
        comp_sum = 0
        for source in comparison_sources:
            if source_list[source - 1]:
                comp_sum += weights[source - 1] * \
                    mag_data[source - 1][image - 1]
        vert_comp_star[image - 1] = comp_sum

    vert_comp_star = vert_comp_star / np.sum(weights, axis=1)

    return vert_comp_star


# In[77]:

"""
Calculate constant that normalizes the set of weights for each of the images (One normalization constant for each image)
This constant will be used to calculate the weights for each source in each image. This weight is calculated by deviding the
sources SNR^2 by the normalization constant for that image.
"""
source_list = [True for el in sources]
source_list[1] = False

comparison_sources = sources  # [4,5,8,9,10]

weights = calc_comp_weights(good_pics, comparison_sources, all_SNR)

vert_comp_star = calc_comp_star(
    good_pics, comparison_sources, corrected_curves, weights)

"""
7/13/2016

The issue you had last night is that the vert_comp_star is varying by many magnitudes, leading you to beleive there
is an error in the way you are calculating the comparison star. Note that the weights ARE all normalized for each
image so that part at least appears to be correct.

Solved
"""

# now we calculate the differential magnitudes
first_differential = np.zeros((len(good_pics), len(sources)))
for source in sources:
    first_differential[:, source -
                       1] = corrected_curves[source - 1] - vert_comp_star

"""
######     ##         #####    ##   ##      ##    ##    ######       ##    ##    ########    ######     ########
#######    ##     #########    ##  ##       ##    ##    #######      ##    ##    ##          #######    ##
##   ##    ##    ###           ## ##        ##    ##    ##   ##      ##    ##    ##          ##   ##    ##
######     ##    ##            ####         ##    ##    ######       ########    #######     ######     #######
##         ##    ###           ####         ###  ###    ##           ##    ##    ##          ##   ##    ##
##         ##     #########    ## ###        ######     ##           ##    ##    ##          ##   ##    ##
##         ##         #####    ##   ##        ####      ##           ##    ##    ########    ##   ##    ########

7/13/2016

You think you got the first differential photometry right, now it's time to make the iterative process.
"""

plt.plot(vert_comp_star, 'o')


# In[ ]:


# In[19]:

"""
Calculate constant that normalizes the set of weights for each of the images (One normalization constant for each image)
This constant will be used to calculate the weights for each source in each image. This weight is calculated by deviding the
sources SNR^2 by the normalization constant for that image.
"""
norm_consts = np.zeros(len(good_pics))
for image in good_pics:
    SNR2sum = 0
    for source in source_index:
        SNR2sum += all_SNR[image - 1, source]**2
    norm_consts[image] = 1 / SNR2sum

# Calculate the initial weights for each source for each image by deviding the sources SNR^2 by the norm const for that image
weights = np.zeros((len(good_pics), len(sources)))
for image, norm in enumerate(norm_consts):
    for source in source_index:
        weights[image, source] = norm * (all_SNR[image, source]**2)

"""
Now the weights should all be calculated... Not too positive on if we should include all sources in this or not but I'm
hoping that the value for the norm constant will eventually make the sources with largest SNR sorta have a small enough
weight that it doesn't matter but we may have to have a larger loop outside of this that takes out certain stars from
the comparison set that are screwing everything up... See figure 1 in Broeg et al. for sorta more clarification (syntax
in that figure is weird).

The weights can now be used to calculate the vertual comparison star
"""

# Make sure that when this is all over every array like corrected_curves is a flat array (if possible).

vert_comp_star = np.zeros(range(goodPics))
for image in good_pics:
    comp_sum = 0
    for source in source_index:
        comp_sum += weights[image, source] * corrected_curves[source][image]
    vert_comp_star[image] = comp_sum

print(vert_comp_star)
lowest_vals = [4]
old_lowest_vals = []
while cmp(lowest_vals, old_lowest_vals) != 0:

    # Calculate comparison
    comp_mags = np.zeros(len(good_picsgood_pics))
    for image in good_picsgood_pics:
        comp_sum = 0
        for obj in lowest_vals:
            comp_sum += (all_SNR[image - 1, obj]**2) * \
                corrected_curves[obj][image]
        comp_mags[image] = comp_sum / norm_consts[image]

    # Calculate differnetial magnitude
    corrected_curves = -1(corrected_curves - comp_mags)

    # Calculate Weights WRONG
    norm_consts = np.zeros(len(good_picsgood_pics))
    for image in good_picsgood_pics:
        SNR2sum = 0
        for obj in lowest_vals:
            SNR2sum += all_SNR[image - 1, obj]**2
        norm_consts[image] = np.std(corrected_curves[obj])

    all_viability = []
    for source in sources:
        new = comp_mags - corrected_curves[source - 1]
        all_viability.append(np.std(new))
    old_lowest_vals, lowest_vals = lowest_vals, list(
        np.argpartition(np.array(all_viability), 5)[0:7])


# In[ ]:


# In[ ]:

#lowest_vals = [23, 35, 40, 25, 27, 29, 34]


# ##Current way of differential photometry is plain wrong.

# In[ ]:

get_ipython().magic(u'matplotlib inline')
star = 1

differential_curve = comp_mags - corrected_curves[source - 1]
for source in sources:
    differential_curve = comp_mags - corrected_curves[source - 1]
    """plt.plot(aij_raw['J.D.-2400000'], corrected_curves[source], 'bo')
    plt.title('Apass calibrated photometry')
    plt.xlabel('J.D.-2400000')
    plt.ylabel('Magnitude')
    #plt.gca().invert_yaxis()
    plt.savefig('Apass calibrated photometry')
    plt.show()"""
    plt.plot(aij_raw['J.D.-2400000'], differential_curve, 'bo')
    plt.title('Apass calibrated auto diff phot ' + str(source))
    plt.xlabel('J.D.-2400000')
    plt.ylabel('Relative Magnitude')
    # plt.gca().invert_yaxis()
    plt.savefig('Apass calibrated automated differential photometry')
    plt.show()


# In[ ]:

table = Table([aij_raw['J.D.-2400000'] + 2400000, differential_curve,
               corrected_curves_er[1]], names=('JD', 'MAG', 'ERROR'))
print(table)


# In[ ]:

table.write('ETD_file.txt', format='ascii.tab')


# In[ ]:


# In[ ]:
