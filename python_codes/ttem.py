import empymod
import numpy as np
import pandas as pd
from scipy.special import roots_legendre
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline
import io

def read_gex_file(fname, n_gates=37):
    fid = io.open(fname, mode="r", encoding="utf-8")
    lines = fid.readlines()
    fid.close()
    lm_waveform = []
    hm_waveform = []
    time_gates = []
    for i_count, line in enumerate(lines):
        if 'TxLoopArea' in line:
            tx_area = float(line.split('=')[-1])
        if 'WaveformLMPoint' in line:
            tmp = line.split()
            lm_waveform.append(np.array([tmp[-2], tmp[-1]], dtype=float))
        if 'WaveformHMPoint' in line:
            tmp = line.split()
            hm_waveform.append(np.array([tmp[-2], tmp[-1]], dtype=float))
        if 'GateTime' in line:
            time_gates.append(line.split('=')[1].split()[-3:])
        if '[Channel1]' in line:
            metadata_ch1 = lines[i_count+1:i_count+17]
        if '[Channel2]' in line:
            metadata_ch2 = lines[i_count+1:i_count+17]
    lm_waveform = np.vstack(lm_waveform)
    hm_waveform = np.vstack(hm_waveform)
    lm_waveform = lm_waveform[np.argwhere(lm_waveform[:,1]==0.)[2][0]:,:]
    hm_waveform = hm_waveform[np.argwhere(hm_waveform[:,1]==0.)[2][0]:,:]
    time_gates = np.vstack(time_gates[:n_gates]).astype(float)
    dict_ch1 = {}
    for dat in metadata_ch1:
        if dat.split() != []:
            tmp = dat.split()[0].split('=')
        try:
            dict_ch1[tmp[0]] = float(tmp[1])
        except:
            dict_ch1[tmp[0]] = tmp[1]
    dict_ch1['waveform'] = lm_waveform
    dict_ch2 = {}
    for dat in metadata_ch2:
        if dat.split() != []:
            tmp = dat.split()[0].split('=')
            try:
                dict_ch2[tmp[0]] = float(tmp[1])
            except:
                dict_ch2[tmp[0]] = tmp[1]
    dict_ch2['waveform'] = hm_waveform
    df_time = pd.DataFrame(data=time_gates, columns=['center', 'start', 'end'])
    output_dict = {'lm': dict_ch1, 'hm': dict_ch2, 'time_gates':df_time, 'tx_area':tx_area}
    return output_dict


def get_ttem_waveform(fname="../python_codes/TX18_60Hz_Calibrated.gex", t0='zero'):
    system = read_gex_file(fname, n_gates=30)
    # SkyTEM ttem
    unit_conversion = 1e-12
    area = system['tx_area']

    i_start_hm = int(system['hm']['RemoveInitialGates'])
    i_start_lm = int(system['lm']['RemoveInitialGates'])
    # i_end_hm = int(system['hm']['RemoveGatesFrom'])
    i_end_lm = int(system['lm']['RemoveGatesFrom'])

    waveform_hm = system['hm']['waveform']
    waveform_lm = system['lm']['waveform']
    time_input_currents_hm = waveform_hm[:,0]
    time_input_currents_lm = waveform_lm[:,0]

    if t0 =='zero':
         time_input_currents_hm -= waveform_hm[:,0].max()
         time_input_currents_lm -= waveform_lm[:,0].max()

    input_currents_hm = waveform_hm[:,1]
    input_currents_lm = waveform_lm[:,1]

    time_gates = system['time_gates'].values
    GateTimeShift=system['lm']['GateTimeShift']
    MeaTimeDelay=system['lm']['MeaTimeDelay']
    NoGates=int(system['lm']['NoGates'])
    t0_lm = waveform_lm[:,0].max()
    times_lm = (time_gates[:NoGates,0] + GateTimeShift + MeaTimeDelay)[i_start_lm:i_end_lm]
    if t0 == 'zero':
        times_lm -= t0_lm
    GateTimeShift=system['hm']['GateTimeShift']
    MeaTimeDelay=system['hm']['MeaTimeDelay']
    NoGates=int(system['hm']['NoGates'])
    t0_hm = waveform_hm[:,0].max()
    times_hm = (time_gates[:NoGates,0] + GateTimeShift + MeaTimeDelay)[i_start_hm:]
    if t0 == 'zero':
        times_hm -= t0_hm
    ttem_waveform = {
        "time_input_currents_lm":time_input_currents_lm,
        "input_currents_lm":input_currents_lm,
        "times_lm":times_lm,
        "time_input_currents_hm":time_input_currents_hm,
        "input_currents_hm":input_currents_hm,
        "times_hm":times_hm,
    }
    return ttem_waveform

def waveform(times, resp, times_wanted, wave_time, wave_amp, nquad=3):
    """Apply a source waveform to the signal.

    Parameters
    ----------
    times : ndarray
        Times of computed input response; should start before and end after
        `times_wanted`.

    resp : ndarray
        EM-response corresponding to `times`.

    times_wanted : ndarray
        Wanted times.

    wave_time : ndarray
        Time steps of the wave.

    wave_amp : ndarray
        Amplitudes of the wave corresponding to `wave_time`, usually
        in the range of [0, 1].

    nquad : int
        Number of Gauss-Legendre points for the integration. Default is 3.

    Returns
    -------
    resp_wanted : ndarray
        EM field for `times_wanted`.

    """

    # Interpolate on log.
    PP = iuSpline(np.log10(times), resp)

    # Wave time steps.
    dt = np.diff(wave_time)
    dI = np.diff(wave_amp)
    dIdt = dI/dt

    # Gauss-Legendre Quadrature; 3 is generally good enough.
    # (Roots/weights could be cached.)
    g_x, g_w = roots_legendre(nquad)

    # Pre-allocate output.
    resp_wanted = np.zeros_like(times_wanted)

    # Loop over wave segments.
    for i, cdIdt in enumerate(dIdt):

        # We only have to consider segments with a change of current.
        if cdIdt == 0.0:
            continue

        # If wanted time is before a wave element, ignore it.
        ind_a = wave_time[i] < times_wanted
        if ind_a.sum() == 0:
            continue

        # If wanted time is within a wave element, we cut the element.
        ind_b = wave_time[i+1] > times_wanted[ind_a]

        # Start and end for this wave-segment for all times.
        ta = times_wanted[ind_a]-wave_time[i]
        tb = times_wanted[ind_a]-wave_time[i+1]
        tb[ind_b] = 0.0  # Cut elements

        # Gauss-Legendre for this wave segment. See
        # https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval
        # for the change of interval, which makes this a bit more complex.
        logt = np.log10(np.outer((tb-ta)/2, g_x)+(ta+tb)[:, None]/2)
        fact = (tb-ta)/2*cdIdt
        resp_wanted[ind_a] += fact*np.sum(np.array(PP(logt)*g_w), axis=1)

    return resp_wanted

def get_time(r_time):
    """Additional time for ramp.

    Because of the arbitrary waveform, we need to compute some times before and
    after the actually wanted times for interpolation of the waveform.

    Some implementation details: The actual times here don't really matter. We
    create a vector of time.size+2, so it is similar to the input times and
    accounts that it will require a bit earlier and a bit later times. Really
    important are only the minimum and maximum times. The Fourier DLF, with
    `pts_per_dec=-1`, computes times from minimum to at least the maximum,
    where the actual spacing is defined by the filter spacing. It subsequently
    interpolates to the wanted times. Afterwards, we interpolate those again to
    compute the actual waveform response.

    Note: We could first call `waveform`, and get the actually required times
          from there. This would make this function obsolete. It would also
          avoid the double interpolation, first in `empymod.model.time` for the
          Fourier DLF with `pts_per_dec=-1`, and second in `waveform`. Doable.
          Probably not or marginally faster. And the code would become much
          less readable.

    Parameters
    ----------
    r_time : ndarray
        Waveform times

    Returns
    -------
    time_req : ndarray
        Required times
    """
    tmin = np.log10(r_time.min())
    tmax = np.log10(r_time.max())
    return np.logspace(tmin, tmax, r_time.size+2)


def ttem_forward_modelling(ttem_waveform, depth, res):
    """Custom wrapper of empymod.model.bipole.

    Here, we compute tTEM data using the ``empymod.model.bipole`` routine as
    an example. We could achieve the same using ``empymod.model.dipole`` or
    ``empymod.model.loop``.

    We model the big source square loop by computing only half of one side of
    the electric square loop and approximating the finite length dipole with 3
    point dipole sources. The result is then multiplied by 8, to account for
    all eight half-sides of the square loop.

    The implementation here assumes a central loop configuration, where the
    receiver (1 m2 area) is at the origin, and the source is a 40x40 m electric
    loop, centered around the origin.

    Note: This approximation of only using half of one of the four sides
          obviously only works for central, horizontal square loops. If your
          loop is arbitrary rotated, then you have to model all four sides of
          the loop and sum it up.


    Parameters
    ----------
    off_time: 
    waveform_times: 
    waveform_current:

    depth : ndarray
        Depths of the resistivity model (see ``empymod.model.bipole`` for more
        info.)

    res : ndarray
        Resistivities of the resistivity model (see ``empymod.model.bipole``
        for more info.)

    Returns
    -------
    tTEM : EMArray
        tTEM response (dB/dt).

    """
    # low moment waveform
    # Get the measurement time and the waveform corresponding to the provided
    # moment.
    # === GET REQUIRED TIMES ===

    time_input_currents_lm = ttem_waveform['time_input_currents_lm']
    input_currents_lm = ttem_waveform['input_currents_lm']
    times_lm = ttem_waveform['times_lm']

    time_input_currents_hm = ttem_waveform['time_input_currents_hm']
    input_currents_hm = ttem_waveform['input_currents_hm']
    times_hm = ttem_waveform['times_hm']

    time = get_time(np.logspace(-6, -2))

    # === GET REQUIRED FREQUENCIES ===
    time, freq, ft, ftarg = empymod.utils.check_time(
        time=time,          # Required times
        signal=-1,           # Switch-on response
        ft='dlf',           # Use DLF
        ftarg={'dlf': 'key_81_CosSin_2009', 'pts_per_dec':-1},  # Short, fast filter; if you
        verb=0,                 # need higher accuracy choose a longer filter.
    )

    # === COMPUTE FREQUENCY-DOMAIN RESPONSE ===
    # We only define a few parameters here. You could extend this for any
    # parameter possible to provide to empymod.model.bipole.
    EM_left = empymod.model.bipole(
        src=[-2, -2, -1, 1, -0.5, -0.5],  # El. bipole source; half of one side.
        rec=[9.28, 9.28, -0.45, 0, 90],         # Receiver at the origin, vertical.
        depth=np.r_[0, depth],        # Depth-model, adding air-interface.
        res=np.r_[2e14, res],         # Provided resistivity model, adding air.
        # aniso=aniso,                # Here you could implement anisotropy...
        #                             # ...or any parameter accepted by bipole.
        freqtime=freq,                # Required frequencies.
        mrec=True,                    # It is an el. source, but a magn. rec.
        strength=1,                   # To account for 1 side of square loop.
        srcpts=3,                     # Approx. the finite dip. with 3 points.
        htarg={'dlf': 'key_101_2009'},  # Short filter, so fast.
        verb=0
    )
    
    EM_right = empymod.model.bipole(
        src=[2, 2, 1, -1, -0.5, -0.5],  # El. bipole source; half of one side.
        rec=[9.28, 9.28, -0.45, 0, 90],         # Receiver at the origin, vertical.
        depth=np.r_[0, depth],        # Depth-model, adding air-interface.
        res=np.r_[2e14, res],         # Provided resistivity model, adding air.
        # aniso=aniso,                # Here you could implement anisotropy...
        #                             # ...or any parameter accepted by bipole.
        freqtime=freq,                # Required frequencies.
        mrec=True,                    # It is an el. source, but a magn. rec.
        strength=1,                   # To account for 1 side of square loop.
        srcpts=3,                     # Approx. the finite dip. with 3 points.
        htarg={'dlf': 'key_101_2009'},  # Short filter, so fast.
        verb=0
    )

    EM_top_bot = empymod.model.bipole(
        src=[-2, 2, 1, 1, -0.5, -0.5],  # El. bipole source; half of one side.
        rec=[9.28, 9.28, -0.45, 0, 90],         # Receiver at the origin, vertical.
        depth=np.r_[0, depth],        # Depth-model, adding air-interface.
        res=np.r_[2e14, res],         # Provided resistivity model, adding air.
        # aniso=aniso,                # Here you could implement anisotropy...
        #                             # ...or any parameter accepted by bipole.
        freqtime=freq,                # Required frequencies.
        mrec=True,                    # It is an el. source, but a magn. rec.
        strength=2,                   # To account for 2 sides of square loop.
        srcpts=3,                     # Approx. the finite dip. with 3 points.
        htarg={'dlf': 'key_101_2009'},  # Short filter, so fast.
        verb=0
    )
    
    # Multiply the frequecny-domain result with
    # \mu for H->B, and i\omega for B->dB/dt.
    EM = EM_left+EM_right+EM_top_bot
    EM *= 2j*np.pi*freq*4e-7*np.pi

    # === Butterworth-type filter (implemented from simpegEM1D.Waveforms.py)===
    # Note: Here we just apply one filter. But it seems that tTEM can apply
    #       two filters, one before and one after the so-called front gate
    #       (which might be related to ``delay_rst``, I am not sure about that
    #       part.)
#     cutofffreq = 4.5e5               # As stated in the tTEM manual
#     h = (1+1j*freq/cutofffreq)**-1   # First order type
#     h *= (1+1j*freq/3e5)**-1
#     EM *= h

    # === CONVERT TO TIME DOMAIN ===
    EM, _ = empymod.model.tem(EM[:, None], np.array([1]), freq, time, 1, ft, ftarg)
    EM = np.squeeze(EM)
    # === APPLY WAVEFORM ===
    data_lm = waveform(time, EM, times_lm, time_input_currents_lm, input_currents_lm)
    data_hm = waveform(time, EM, times_hm, time_input_currents_hm, input_currents_hm)
    tx_area = 4*2
    return data_lm/tx_area, data_hm/tx_area