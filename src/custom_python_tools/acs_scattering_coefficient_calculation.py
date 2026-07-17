from scipy.interpolate import interp1d

def calculate_scattering_spectrum(absorption_spectrum, wav_A, beam_attenuation_spectrum, wav_C):
    """ 
    Interpolates Beam Attenuation (C) to Absorption (A) wavelengths 
    and subtracts them to find Scattering (B).

    Args:
        absorption_spectrum (array): 
        wav_A (array): the wavelength of the absorption spectrum.
        beam_attenuation_spectrum (array): 
        wav_C (array): the wavelength of the beam attenuation spectrum.

    Returns:
        scattering_spectrum (spectrum) at the same wavelengths as the absorption spectrum.
    """
   
    # Create the interpolation function for C
    f_interp_c = interp1d(wav_C.astype(float), beam_attenuation_spectrum, kind='linear', fill_value="extrapolate")
    
    # Get C values at A wavelengths
    c_interpolated = f_interp_c(wav_A.astype(float))
    
    # b = c - a
    scattering_spectrum = c_interpolated - absorption_spectrum
    
    return scattering_spectrum