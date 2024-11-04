import numpy as np
from numpy import log
from scipy.optimize import curve_fit
from unyt import c

def em_to_obs(em, z):
    """
    Convert from emitted to observed wavelength for a given redshift.
    Args:
        em: emitted wavelength
        z: redshift
    Returns:
        obs: observed wavelength
    """
    if type(em) == list: #cant multiply list by a float typically
        return (1.+z)*np.array(em)
    return (1.+z)*em

def obs_to_em(obs, z):
    """
    Convert from observed to emitted wavelength for a given redshift.
    Args:
        obs: observed wavelength
        z: redshift
    Returns:
        em: emitted wavelength
    """
    if type(obs) == list: #cant multiply list by a float typically
        return np.array(obs)/(1.+z)
    return obs/(1.+z)

def flam_model(wavelength, normalisation, beta):
    """Power-law model for flam."""
    return  normalisation * (wavelength) ** beta

def fnu_model(wavelength, normalisation, beta):
    """Power-law model for fnu."""
    return  normalisation * (wavelength) ** (beta +2.)

def linearmodel(x, m, c):
    """Linear model."""
    return m*x + c

def spectral_fit(lower, upper, z, lam, flam, flam_e):
    """
    Performs a spectral fit of flam from lower to upper.
    Performs scaled fits if default fit isn't reliable.
    Args:
        lower: lower rest-frame wavelength range
        upper: upper rest-frame wavelength range
        z: redshift
        lam: wavelength array (AA)
        flam: flux array (erg/s/cm**2/AA)
        flam_e: flux error array (erg/s/cm**2/AA)
    Returns:
        beta, beta_error if fitted: 2222 if unable to fit, 3333 if fits disagree
    """
    lam_range, flam_range, flam_e_range = return_within_range(lower, upper, z, lam, flam, flam_e)    
    try:
        # FIT DEFAULT SPECTRUM
        popt, pcov = curve_fit(flam_model, lam_range, flam_range, p0 = [1.,-2.], sigma = flam_e_range, absolute_sigma = True, bounds = ((-np.inf,-10), (np.inf,10)))
        defaultbeta, defaulterror = popt[1], np.sqrt(np.diag(pcov))[1]
        if defaulterror <= 1e-15:
            # IF FIT GIVES UNREASONABLE ERROR TRY DIFFERENTLY SCALED FITS
            # SINCE WE CARE ABOUT THE SLOPE AND A UNIFORM SCALE KEEPS THE SHAPE, ABLE TO DO THIS

            # SCALED FIT IS SIMPLY BY SCALING FLAM BY 1e15
            popt, pcov = curve_fit(flam_model, lam_range, flam_range*1e15, p0 = [1.,-2.], sigma = flam_e_range*1e15, absolute_sigma = True, bounds = ((-np.inf,-10), (np.inf,10)))
            scaledbeta, scalederror = popt[1], np.sqrt(np.diag(pcov))[1]
            if abs(scaledbeta - defaultbeta) <= 0.01 and scalederror != 0.0 and scalederror > 1e-15:
                # IF SCALED FIT AGREES WITH DEFAULT, AND ERROR IS REASONABLE THEN USE IT
                return scaledbeta, scalederror
            
            # INDEX FIT IS SCALED BY THE VALUE OF FLAM AT 3500 A
            # FOR 4 SPECTRA IN DR3 THIS VALUE IS NEGATIVE, SO USE ABSOLUTE VALUE, ULTIMATELY DOESN'T REALLY MATTER AS LONG AS IT IS SAME ROUGH SIZE AS COLLECTED FLAM
            index3500 = min(range(len(lam)), key=lambda i: abs(lam[i]-em_to_obs(3500,z)))
            
            
            popt, pcov = curve_fit(flam_model, lam_range, flam_range/abs(flam[index3500]), p0 = [1.,-2.], sigma = flam_e_range/abs(flam_e[index3500]), absolute_sigma = True, bounds = ((-np.inf,-10), (np.inf,10)))
            indexbeta, indexerror = popt[1], np.sqrt(np.diag(pcov))[1]
            if abs(indexbeta - defaultbeta) <= 0.01 and indexerror != 0.0 and indexerror > 1e-15:
                # IF INDEX FIT AGREES WITH DEFAULT, AND ERROR IS REASONABLE THEN USE IT
                return indexbeta, indexerror

            # IF NEITHER SCALED OR INDEX AGREE WITH DEFAULT, CHECK IF THEY AGREE WITH EACH OTHER; ONLY OCCURS ONCE FOR 1300-3500 JADES DR3
            if abs(indexbeta - scaledbeta) <= 0.01 and scalederror != 0.0 and scalederror > 1e-15:
                # IF THEY AGREE, USE SCALED
                return scaledbeta, scalederror

            # IF NO VALUES AGREE WITH EACH OTHER THEN UNABLE TO FIT
            # print("DISAGREE SPECTRAL")
            # print(defaultbeta, defaulterror)
            # print(scaledbeta, scalederror)
            # print(indexbeta, indexerror)
            return 3333, 3333
        # IF DEFAULT ERROR IS REASONABLE, RETURN DEFAULT FIT
        return popt[1], np.sqrt(np.diag(pcov))[1]
    except RuntimeError:
        return 2222, 2222

def spectral_calzetti(z, lam, flam, flam_e):
    """
    Performs a spectral fit of flam using Calzetti windows.
    Performs scaled fits if default fit isn't reliable.
    Args:
        z: redshift
        lam: wavelength array (AA)
        flam: flux array (erg/s/cm**2/AA)
        flam_e: flux error array (erg/s/cm**2/AA)
    Returns:
        beta, beta_error if fitted: 2222 if unable to fit, 3333 if fits disagree
    """

    # CONVERT LAM TO EMITTED TO MATCH WINDOWS, AND CONVERT BACK AFTERWARDS
    calzetti_lam = [x for x in obs_to_em(lam, z) if (x >= 1268. and x <= 1284.) or (x >= 1309. and x <= 1316.) or (x >= 1342. and x <= 1371.) or (x >= 1407. and x <= 1515.) or (x >= 1562. and x <= 1583.) or (x >= 1677. and x <= 1740.) or (x >= 1760. and x <= 1833.) or (x >= 1866. and x <= 1890.) or (x >= 1930. and x <= 1950.) or (x >= 2400. and x <= 2580.)]
    calzetti_lam = em_to_obs(calzetti_lam, z)

    calzetti_flam = [y for x, y in zip(obs_to_em(lam, z), flam) if (x >= 1268. and x <= 1284.) or (x >= 1309. and x <= 1316.) or (x >= 1342. and x <= 1371.) or (x >= 1407. and x <= 1515.) or (x >= 1562. and x <= 1583.) or (x >= 1677. and x <= 1740.) or (x >= 1760. and x <= 1833.) or (x >= 1866. and x <= 1890.) or (x >= 1930. and x <= 1950.) or (x >= 2400. and x <= 2580.)]
    calzetti_flam_e = [y for x, y in zip(obs_to_em(lam, z), flam_e) if (x >= 1268. and x <= 1284.) or (x >= 1309. and x <= 1316.) or (x >= 1342. and x <= 1371.) or (x >= 1407. and x <= 1515.) or (x >= 1562. and x <= 1583.) or (x >= 1677. and x <= 1740.) or (x >= 1760. and x <= 1833.) or (x >= 1866. and x <= 1890.) or (x >= 1930. and x <= 1950.) or (x >= 2400. and x <= 2580.)]
    
    calzetti_lam = np.array(calzetti_lam)
    calzetti_flam = np.array(calzetti_flam)
    calzetti_flam_e = np.array(calzetti_flam_e)

    try:
        popt, pcov = curve_fit(flam_model, calzetti_lam, calzetti_flam, p0 = [1.,-2.], sigma = calzetti_flam_e, absolute_sigma = True, bounds = ((-np.inf,-10), (np.inf,10)))
        defaultbeta, defaulterror = popt[1], np.sqrt(np.diag(pcov))[1]
        if defaulterror <= 1e-15:
            # IF FIT GIVES UNREASONABLE ERROR TRY DIFFERENTLY SCALED FITS
            # SINCE WE CARE ABOUT THE SLOPE AND A UNIFORM SCALE KEEPS THE SHAPE, ABLE TO DO THIS

            # SCALED FIT IS SIMPLY BY SCALING FLAM BY 1e15
            popt, pcov = curve_fit(flam_model, calzetti_lam, calzetti_flam*1e15, p0 = [1.,-2.], sigma = calzetti_flam_e*1e15, absolute_sigma = True, bounds = ((-np.inf,-10), (np.inf,10)))
            scaledbeta, scalederror = popt[1], np.sqrt(np.diag(pcov))[1]
            if abs(scaledbeta - defaultbeta) <= 0.01 and scalederror != 0.0 and scalederror > 1e-15:
                # IF SCALED FIT AGREES WITH DEFAULT, AND ERROR IS REASONABLE THEN USE IT
                return scaledbeta, scalederror
            
            # INDEX FIT IS SCALED BY THE VALUE OF FLAM AT 3500 A
            # FOR 4 SPECTRA IN DR3 THIS VALUE IS NEGATIVE, SO USE ABSOLUTE VALUE, ULTIMATELY DOESN'T REALLY MATTER AS LONG AS IT IS SAME ROUGH SIZE AS COLLECTED FLAM
            index3500 = min(range(len(lam)), key=lambda i: abs(lam[i]-em_to_obs(3500,z)))
            
            
            popt, pcov = curve_fit(flam_model, calzetti_lam, calzetti_flam/abs(flam[index3500]), p0 = [1.,-2.], sigma = calzetti_flam_e/abs(flam_e[index3500]), absolute_sigma = True, bounds = ((-np.inf,-10), (np.inf,10)))
            indexbeta, indexerror = popt[1], np.sqrt(np.diag(pcov))[1]
            if abs(indexbeta - defaultbeta) <= 0.01 and indexerror != 0.0 and indexerror > 1e-15:
                # IF INDEX FIT AGREES WITH DEFAULT, AND ERROR IS REASONABLE THEN USE IT
                return indexbeta, indexerror

            # IF NEITHER SCALED OR INDEX AGREE WITH DEFAULT, CHECK IF THEY AGREE WITH EACH OTHER; ONLY OCCURS ONCE FOR 1300-3500 JADES DR3
            if abs(indexbeta - scaledbeta) <= 0.01 and scalederror != 0.0 and scalederror > 1e-15:
                # IF THEY AGREE, USE SCALED
                return scaledbeta, scalederror

            # IF NO VALUES AGREE WITH EACH OTHER THEN UNABLE TO FIT
            # print("DISAGREE CALZETTI")
            # print(defaultbeta, defaulterror)
            # print(scaledbeta, scalederror)
            # print(indexbeta, indexerror)
            return 3333, 3333
        # IF DEFAULT ERROR IS REASONABLE, RETURN DEFAULT FIT
        return popt[1], np.sqrt(np.diag(pcov))[1]
    except RuntimeError:
        return 2222, 2222

def return_within_range(lower, upper, z, lam, flam = None, flam_e = None):
    """
    Return arrays within lower, upper wavelength range.
    Args:
        lower: lower rest-frame wavelength limit (AA)
        upper: upper rest-frame wavelegnth limit (AA)
        z: redshift
        lam: wavelength array (AA)
        flam: flux array (erg/s/cm**2/AA)
        flam_e: flux error array (erg/s/cm**2/AA)
    Returns:
        If flam_e is given returns lam, flam and flam_e within wavelength range.
        If flam is given but not flam_e returns lam and flam within range.
        If only lam is given returns lam within range.
    """
    lam_within_range = [x for x in lam if x > em_to_obs(lower, z) and x < em_to_obs(upper, z)]

    if flam_e is not None: #Assumes if flam_e is given, then flam will also be given
        flam_within_range = [y for x, y in zip(lam, flam) if x > em_to_obs(lower, z) and x < em_to_obs(upper, z)]
        flam_e_within_range = [y for x, y in zip(lam, flam_e) if x > em_to_obs(lower, z) and x < em_to_obs(upper, z)]
        return np.array(lam_within_range), np.array(flam_within_range), np.array(flam_e_within_range)
    
    if flam is not None:
        flam_within_range = [y for x, y in zip(lam, flam) if x > em_to_obs(lower, z) and x < em_to_obs(upper, z)]
        return np.array(lam_within_range), np.array(flam_within_range)

    return np.array(lam_within_range)

def spectral_fit_integration(lower, upper, z, lam, flam):
    """
    Performs a spectral fit over two colours, similar to FLARES method, centred at lower and upper, +- 200 AA.
    Args:
        lower: lower rest-frame wavelength limit (AA)
        upper: upper rest-frame wavelegnth limit (AA)
        z: redshift
        lam: wavelength array (AA)
        flam: flux array (erg/s/cm**2/AA)
    Returns:
        beta
    """
    blue_lam, blue_flam = return_within_range(lower - 200., lower + 200., z, lam, flam)
    red_lam, red_flam = return_within_range(upper - 200., upper + 200., z, lam, flam)

    integratedblueflam = abs(np.trapz(blue_flam, blue_lam))
    integratedredflam = abs(np.trapz(red_flam, red_lam))

    return log(integratedblueflam/integratedredflam)/log(lower/upper)

    # From previous time when comparing spectral vs integration
    # lam_range = blue_lam + red_lam
    # flam_range = blue_flam + red_flam
    # flam_e_range = blue_flam_e + red_flam_e
    # return blue_lam, blue_flam, blue_flam_e, red_lam, red_flam, red_flam_e

def absolute_mag_to_log10lnu(ab_mag):
    """
    Convert between absolute magnitude (AB) and log10lnu.
    Args:
        ab_mag: Absolute magnitude
    Returns:
        log10lnu
    """
    return np.log10(10 ** (-0.4 * (ab_mag + 48.6)) * 1.1964951728479152e+40)

def log10lnu_to_absolute_mag(log10lnu):
    """
    Convert between log10lnu and absolute magnitude (AB).
    Args:
        log10lnu: log (base 10) of spectral luminosity (in erg / s / Hz)
    Returns:
        ab_mag
    """
    return -2.5 * np.log10(10**log10lnu / 1.1964951728479152e+40) - 48.6

def flam_to_fnu(lam, flam):
    """
    Originally from synthesizer
    Converts spectral flux in terms of wavelength (f_lam) to spectral flux
    in terms of frequency (f_nu).

    Args:
        lam (unyt_quantity/unyt_array)
            The wavelength array the flux is defined at.
        flam (unyt_quantity/unyt_array)
            The spectral flux in terms of wavelength.

    Returns:
        unyt_quantity/unyt_array
            The spectral flux in terms of frequency, in units of nJy.

    Raises:
        IncorrectUnits
            If units are missing an error is raised.
    """


    return (flam * lam**2 / c).to("nJy")








# DIFFERENT SPECTRAL FITS TO TEST COMPARISON
def spectral_fit_LMnoerror(lower, upper, lam, flam, z):
    """
    Performs a spectral fit using the LM method with no errors provided.
    \nSee spectral_fit for explanation of arguments.
    """
    lam_range, flam_range = return_within_range(lower, upper, z, lam, flam)
    try:
        popt, pcov = curve_fit(flam_model, lam_range, flam_range, p0 = [1.,-2.], method = "lm")
        return popt[1], np.sqrt(np.diag(pcov))[1]
    except RuntimeError:
        return 2222, 2222

def spectral_fit_LMwitherror(lower, upper, lam, flam, flam_e, z):
    """
    Performs a spectral fit using the LM method and errors.
    \nSee spectral_fit for explanation of arguments.
    """
    lam_range, flam_range, flam_e_range = return_within_range(lower, upper, z, lam, flam, flam_e)
    try:
        popt, pcov = curve_fit(flam_model, lam_range, flam_range, p0 = [1.,-2.], sigma = flam_e_range, absolute_sigma = True, method = "lm")
        return popt[1], np.sqrt(np.diag(pcov))[1]
    except RuntimeError:
        return 2222, 2222

def spectral_fit_TRFnoerror(lower, upper, lam, flam, z):
    """
    Performs a spectral fit using the TRF method with no errors provided.
    \nSee spectral_fit for explanation of arguments.
    """
    lam_range, flam_range = return_within_range(lower, upper, z, lam, flam)
    try:
        popt, pcov = curve_fit(flam_model, lam_range, flam_range, p0 = [1.,-2.], bounds = ((-np.inf,-10), (np.inf,10)), method = "trf")
        return popt[1], np.sqrt(np.diag(pcov))[1]
    except RuntimeError:
        return 2222, 2222

def spectral_fit_TRFwitherror(lower, upper, lam, flam, flam_e, z):
    """
    Performs a spectral fit using the TRF method and errors.
    \nSee spectral_fit for explanation of arguments.
    """
    lam_range, flam_range, flam_e_range = return_within_range(lower, upper, z, lam, flam, flam_e)
    try:
        popt, pcov = curve_fit(flam_model, lam_range, flam_range, p0 = [1.,-2.], sigma = flam_e_range, absolute_sigma = True, bounds = ((-np.inf,-10), (np.inf,10)), method = "trf")
        return popt[1], np.sqrt(np.diag(pcov))[1]
    except RuntimeError:
        return 2222, 2222


# #OLD FUNCTIONS:
# def nearest_filter(z, wavelength = 1500):#M filters differ between DR2 and DR3- could do coord check? currently only does W
#     """
#     Find NIRCam filter closest to selected wavelength, at a specified wavelength.
#     \nArgs:
#         z: Redshift
#         wavelength: Wavelength we want to get closest to (in angstrom) - default is 1500
#     \nReturns:
#         filter: Closest filter to target wavelength.
#     """
#     observed = em_to_obs(wavelength, z)
#     wide_filters = [11500,15000,20000,27700,35600,44400] # ignores F090W since its the default/minimum scenario
#     difference = abs(observed - 9000)
#     best_filter = 9000
#     for filter in wide_filters:
#         if abs(observed - filter) < difference:
#             difference = abs(observed - filter)
#             best_filter = filter
#     return f"F{int(best_filter/100):03}W"

# def spectral_calzetti_interp(z, lam, flam):
#     """
#     Performs a spectral fit of flam using Calzetti windows and interpolating values.
#     \nSince interpolating, doesn't currently take errors into account, nor does it check differently scaled fits like other spectral fits.
#     """
#     lower_calzetti = em_to_obs(np.array([1268., 1309., 1342., 1407., 1562., 1677., 1760., 1866., 1930., 2400.]), z)# * angstrom
#     upper_calzetti = em_to_obs(np.array([1284., 1316., 1371., 1515., 1583., 1740., 1833., 1890., 1950., 2580.]), z)# * angstrom
#     calzetti_lam = (lower_calzetti+upper_calzetti)/2
#     calzetti_flam = []
#     for lower, upper in zip(lower_calzetti, upper_calzetti):
#         # finds index of lam just below lower - default is list so get value with [0]
#         # interpolate between lower and lower+1 to get interp at lower
#         lowerminus1 = np.argwhere(np.array(lam)<=lower)[-1][0]
#         lower_interp = np.interp(lower, lam[lowerminus1:lowerminus1+2], flam[lowerminus1:lowerminus1+2])# * erg/s/cm**2/angstrom

        
#         upperminus1 = np.argwhere(np.array(lam)<=upper)[-1][0]
#         upper_interp = np.interp(upper, lam[upperminus1:upperminus1+2], flam[upperminus1:upperminus1+2])# * erg/s/cm**2/angstrom

#         interp_lam = [lower, upper]
#         interp_flam = [lower_interp, upper_interp]

#         within_window_lam = [x for x in lam if x >= lower and x <= upper]
#         within_window_flam = [y for x,y in zip(lam,flam) if x >= lower and x <= upper]

#         integrateable_lam = interp_lam + within_window_lam
#         integrateable_flam = interp_flam + within_window_flam
#         #need to rewrite
#         sorted_lam, sorted_flam = (list(t) for t in zip(*sorted(zip(integrateable_lam, integrateable_flam))))
#         calzetti_flam.append(abs(np.trapz(sorted_flam, sorted_lam))/(upper-lower))

#     try:
#         popt, pcov = curve_fit(flam_model, calzetti_lam, calzetti_flam, p0 = [1.,-2.])
#         return popt[1], np.sqrt(pcov[1][1])
#     except RuntimeError:
#         return 2222, 2222