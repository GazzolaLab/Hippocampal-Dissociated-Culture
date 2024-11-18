import numpy as np
import yaml
from copy import deepcopy
from collections import defaultdict
import random
from neuron import h
from scipy.interpolate import Akima1DInterpolator
import logging


def place_field_fr(center, spatial_bins, max_fr, diameter):
    c = diameter / 4.3
    fnc = max_fr * np.exp(-(((spatial_bins - center) / (c)) ** 2.0))
    return fnc


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_inhom_poisson_spike_times_by_thinning(
    rate, t, dt=0.02, delay=0.0, refractory=3.0, generator=None
):
    """
    Given a time series of instantaneous spike rates in Hz, produce a spike train consistent with an inhomogeneous
    Poisson process with a refractory period after each spike.
    :param rate: instantaneous rates in time (Hz)
    :param t: corresponding time values (ms)
    :param dt: temporal resolution for spike times (ms)
    :param refractory: absolute deadtime following a spike (ms)
    :param generator: :class:'np.random.RandomState()'
    :return: list of m spike times (ms)
    """
    tt = deepcopy(t)
    #     fr_dt = tt[1]-tt[0]
    #     tt += delay
    #     expanded_t    = np.arange(0,delay,step=fr_dt)
    #     expanded_rate = [1.0 for _ in range(len(expanded_t))]
    #     rate = np.asarray(expanded_rate + list(rate), dtype='float32')
    #     tt    = np.asarray(list(expanded_t) + list(tt), dtype='float32')
    min_fr = np.min(rate)
    if generator is None:
        generator = random
    interp_t = np.arange(tt[0], tt[-1], dt)

    try:
        rate_ip = Akima1DInterpolator(tt, rate)
        interp_rate = rate_ip(interp_t)
    except Exception:
        print("t shape: %s rate shape: %s" % (str(tt.shape), str(rate.shape)))
        raise

    delay_t = np.arange(-delay, 0, dt)
    delay_r = np.ones(len(delay_t)) * 2.0

    interp_t = np.concatenate((delay_t, interp_t))
    interp_t += delay
    interp_rate = np.concatenate((delay_r, interp_rate))

    interp_rate /= 1000.0
    spike_times = []
    non_zero = np.where(interp_rate > 1.0e-100)[0]
    if len(non_zero) == 0:
        return spike_times
    interp_rate[non_zero] = 1.0 / (1.0 / interp_rate[non_zero] - refractory)
    max_rate = np.max(interp_rate)
    if not max_rate > 0.0:
        return spike_times
    i = 0
    ISI_memory = 0.0
    while i < len(interp_t):
        x = generator.uniform(0.0, 1.0)
        if x > 0.0:
            ISI = -np.log(x) / max_rate
            i += int(ISI // dt)
            ISI_memory += ISI
            if (
                (i < len(interp_t))
                and (generator.uniform(0.0, 1.0) <= interp_rate[i] / max_rate)
                and ISI_memory >= 0.0
            ):
                spike_times.append(interp_t[i])
                ISI_memory = -refractory
    return np.asarray(spike_times, dtype="float32")


class SpatialFeatureSpace(object):
    def __init__(self, params_filepath, super_arena_flanks=(0, 0), theta_modulation={}):
        self.pc = h.ParallelContext()
        self.params = {}
        self.params_filepath = params_filepath
        self._read_arena_params()
        self._read_arena_cell_params()

        arena_rnd = None
        if int(self.pc.id()) == 0:
            arena_rnd = np.random.RandomState(seed=self.params["Arena"]["random seed"])
        self.arena_rnd = self.pc.py_broadcast(arena_rnd, 0)

        self.arena_size = self.params["Arena"]["arena size"]
        self.bin_size = self.params["Arena"]["bin size"]
        self.arena_map = np.arange(0, self.arena_size, step=self.bin_size)
        self.cell_information = {}

    def _read_arena_params(self):
        self.params["Arena"] = {}
        arena_params = None
        if self.pc.id() == 0:
            with open(self.params_filepath, "r") as f:
                fparams = yaml.load(f, Loader=yaml.FullLoader)
                arena_params = fparams["Arena"]
                self.params["Arena"] = arena_params
        self.pc.barrier()
        self.params["Arena"] = self.pc.py_broadcast(arena_params, 0)

    def _read_arena_cell_params(self):
        self.params["Spatial"] = {}
        arena_cell_params = None
        if self.pc.id() == 0:
            with open(self.params_filepath, "r") as f:
                fparams = yaml.load(f, Loader=yaml.FullLoader)
                arena_cell_params = fparams["Arena Cells"]
                self.params["Spatial"] = arena_cell_params
        self.pc.barrier()
        self.params["Spatial"] = self.pc.py_broadcast(arena_cell_params, 0)

    def generate_population_firing_rates(self, seed=2e9):
        for population_name in self.params["Spatial"].keys():
            self.cell_information[population_name] = {}
            current_population = self.params["Spatial"][population_name]

            self.cell_information[population_name]["id"] = current_population["id"]
            self.cell_information[population_name]["ncells"] = current_population[
                "ncells"
            ]
            somatic_positions = generate_soma_positions(current_population["ncells"])

            ncells = current_population["ncells"]
            self.cell_information[population_name]["cell info"] = {}
            ctype_offset = 0

            for idx in range(ncells):
                gid = idx + ctype_offset
                self.cell_information[population_name]["cell info"][gid] = {}
                self.cell_information[population_name]["cell info"][gid][
                    "soma position"
                ] = somatic_positions[idx]

            if "place" in current_population:
                self.cell_information[population_name]["spatial type"] = "place"
                field_centers = soma_positions_to_field_center(
                    somatic_positions, self.arena_size
                )
                for idx in range(ncells):
                    gid = idx + ctype_offset
                    self.cell_information[population_name]["cell info"][gid][
                        "field center"
                    ] = field_centers[idx]

                peak_rates = current_population["place"]["peak rates"]
                peak_rate_probs = current_population["place"]["peak rate probabilities"]
                peak_rate_prob_sum = np.sum(peak_rate_probs)
                peak_rate_probs_nrm = peak_rate_probs / peak_rate_prob_sum
                min_rate = current_population["place"]["min rate"]
                diameter = current_population["place"]["diameter"]
                place_firing_rates = generate_place_firing_maps(
                    field_centers,
                    peak_rates,
                    peak_rate_probs_nrm,
                    min_rate,
                    diameter,
                    self.arena_map,
                    self.arena_rnd,
                )
                for idx in range(ncells):
                    gid = idx + ctype_offset
                    self.cell_information[population_name]["cell info"][gid][
                        "firing rate"
                    ] = place_firing_rates[idx]

            elif "grid" in current_population:
                self.cell_information[population_name]["spatial type"] = "grid"
                field_centers = soma_positions_to_field_center(
                    somatic_positions, self.arena_size
                )
                for idx in range(ncells):
                    gid = idx + ctype_offset
                    self.cell_information[population_name]["cell info"][gid][
                        "field center"
                    ] = field_centers[idx]

                peak_rates = current_population["grid"]["peak rates"]
                peak_rate_probs = current_population["grid"]["peak rate probabilities"]
                peak_rate_prob_sum = np.sum(peak_rate_probs)
                peak_rate_probs_nrm = peak_rate_probs / peak_rate_prob_sum
                min_rate = current_population["grid"]["min rate"]
                diameter = current_population["grid"]["diameter"]
                gap = current_population["grid"]["gap"]

                grid_firing_rates = generate_grid_firing_maps(
                    field_centers,
                    peak_rates,
                    peak_rate_probs_nrm,
                    min_rate,
                    diameter,
                    gap,
                    self.arena_map,
                    self.arena_rnd,
                )
                for idx in range(ncells):
                    gid = idx + ctype_offset
                    self.cell_information[population_name]["cell info"][gid][
                        "firing rate"
                    ] = grid_firing_rates[idx]

            ctype_offset += ncells

    def generate_cue_firing_rates(self, population, percent_cue):
        noise_fr = self.params["Spatial"][population]["noise"]["mean rate"]
        ncells = self.params["Spatial"][population]["ncells"]

        ncue_cells = int(ncells * percent_cue)
        cells_cued = self.arena_rnd.choice(
            np.arange(ncells), size=(ncue_cells,), replace=False
        )
        min_fr = self.params["Spatial"][population]["cue"]["min rate"]
        peak_rates = self.params["Spatial"][population]["cue"]["peak rates"]
        peak_rate_probs = self.params["Spatial"][population]["cue"][
            "peak rate probabilities"
        ]
        peak_rate_prob_sum = np.sum(peak_rate_probs)
        peak_rate_probs_nrm = peak_rate_probs / peak_rate_prob_sum
        diameter = self.params["Spatial"][population]["cue"]["diameter"]
        cue_firing_rates = []
        for i in range(ncells):
            cue_fr = None
            if i in cells_cued:
                max_fr = self.arena_rnd.choice(peak_rates, p=peak_rate_probs_nrm)
                cue_fr = place_field_fr(
                    int(self.arena_size / 2), self.arena_map, max_fr, diameter
                )
            else:
                cue_fr = [noise_fr for _ in range(len(self.arena_map))]
            cue_fr = np.asarray(cue_fr, dtype="float32")
            cue_fr[cue_fr <= min_fr] = min_fr
            cue_firing_rates.append(cue_fr)
        self.cell_information[population] = {}
        self.cell_information[population]["ncells"] = ncells
        self.cell_information[population]["id"] = self.params["Spatial"][population][
            "id"
        ]
        self.cell_information[population]["cell info"] = {}
        for idx, cfr in enumerate(cue_firing_rates):
            self.cell_information[population]["cell info"][idx] = {}
            self.cell_information[population]["cell info"][idx]["firing rate"] = cfr

    def generate_spike_times(self, population, dt=0.05, delay=0, cued=False):
        mouse_speed = self.params["Arena"]["mouse speed"]
        lap_information = self.params["Arena"]["lap information"]
        nlaps = lap_information["nlaps"]
        is_spatial = lap_information["is spatial"]
        up_state = lap_information.get("up state", None)
        run_step_dur = self.bin_size / mouse_speed
        arena_length = len(self.arena_map)
        start_time = 0
        end_time = nlaps * self.arena_size / mouse_speed
        bin2times = np.arange(0, end_time, step=run_step_dur, dtype="float32") * 1000.0

        population_info = self.cell_information[population]
        ncells = population_info["ncells"]
        gids = range(int(self.pc.id()), ncells, int(self.pc.nhost()))

        nfr = np.clip(
            self.arena_rnd.normal(
                self.params["Spatial"][population]["noise"]["mean rate"],
                scale=self.params["Spatial"][population]["noise"]["scale"],
                size=ncells,
            ),
            0.0,
            None,
        )
        noise_fr = np.vstack([nfr for _ in range(arena_length)])

        up_state_fr = None
        if "up state" in self.params["Spatial"][population]:
            usfr = np.clip(
                self.arena_rnd.normal(
                    self.params["Spatial"][population]["up state"]["mean rate"],
                    scale=self.params["Spatial"][population]["up state"]["scale"],
                    size=ncells,
                ),
                0.0,
                None,
            )
            usfr_on_duration = self.params["Spatial"][population]["up state"].get(
                "on duration", None
            )
            usfr_off_duration = self.params["Spatial"][population]["up state"].get(
                "off duration", None
            )
            if usfr_off_duration is None:
                usfr_off_duration = usfr_on_duration
            up_state_fr = None
            if usfr_on_duration is not None:
                up_state_on_length = int(round(usfr_on_duration / run_step_dur))
                up_state_off_length = int(round(usfr_off_duration / run_step_dur))
                n_up_state_fr = 0
                up_state_fr_list = []
                while n_up_state_fr < arena_length:
                    up_state_fr_list.extend([usfr for _ in range(up_state_on_length)])
                    n_up_state_fr += up_state_on_length
                    up_state_fr_list.extend([nfr for _ in range(up_state_off_length)])
                    n_up_state_fr += up_state_off_length
                up_state_fr = np.vstack(up_state_fr_list[:arena_length])
            else:
                up_state_fr = np.vstack([usfr for _ in range(arena_length)])

        firing_rates = {}
        for gid in gids:
            try:
                fr = population_info["cell info"][gid]["firing rate"]
            except:
                fr = noise_fr[:, gid]
            firing_rates[gid] = fr

        # self.arena_size, self.bin_size

        if cued:
            self.cued_positions = np.linspace(
                12.5, self.arena_size - 12.5, np.sum(is_spatial)
            )
            self.random_cue_locs = np.arange(len(self.cued_positions))
            self.arena_rnd.shuffle(self.random_cue_locs)
            print(self.random_cue_locs)
        for gid, fr in firing_rates.items():
            current_full_fr = []
            online_number = 0
            for n in range(nlaps):
                if not is_spatial[n]:
                    if (
                        (up_state_fr is not None)
                        and (up_state is not None)
                        and (up_state[n] > 0)
                    ):
                        this_fr = up_state_fr[:, gid]

                        current_full_fr.extend(this_fr)
                    else:
                        if population == "MF":
                            current_full_fr.extend(np.multiply(noise_fr[:, gid], 1.0))
                        else:
                            current_full_fr.extend(noise_fr[:, gid])
                else:
                    if cued:
                        random_position = self.cued_positions[
                            self.random_cue_locs[online_number]
                        ]
                        to_roll = int(
                            (self.arena_size / 2 - random_position)
                            / (self.arena_map[1] - self.arena_map[0])
                        )
                        current_full_fr.extend(np.roll(fr, to_roll))
                    else:
                        current_full_fr.extend(fr)
                    online_number += 1
            current_full_fr = np.asarray(current_full_fr, dtype="float32")
            if bin2times.shape[0] > current_full_fr.shape[0]:
                bin2times = bin2times[:-1]
            spike_times = np.asarray(
                get_inhom_poisson_spike_times_by_thinning(
                    current_full_fr, bin2times, dt=dt, delay=delay
                ),
                dtype="float32",
            )
            self.cell_information[population]["cell info"][gid][
                "spike times"
            ] = spike_times


def generate_soma_positions(ncells, maxpos=1.0):
    positions = np.linspace(0, maxpos, ncells)
    return {i: positions[i] for i in range(ncells)}


def soma_positions_to_field_center(volume_positions, arena_size, maxpos=1.0):
    positions = np.linspace(0, maxpos, 100)
    centers = np.linspace(0, arena_size, 100)
    spatial_ip = Akima1DInterpolator(positions, centers)

    spatial_positions = {}
    for gid in volume_positions.keys():
        gid_pos = volume_positions[gid]
        spatial_positions[gid] = spatial_ip(gid_pos)
    return spatial_positions


def generate_place_firing_maps(
    field_centers, peak_rates, peak_rate_probs, min_rate, diameter, arena_map, rnd
):
    firing_rates = {}
    for gid in list(field_centers.keys()):
        max_rate = rnd.choice(peak_rates, p=peak_rate_probs)
        current_center = field_centers[gid]
        fr = np.asarray(
            place_field_fr(current_center, arena_map, max_rate, diameter),
            dtype="float32",
        )
        fr[fr <= min_rate] = min_rate
        firing_rates[gid] = fr
    return firing_rates


def generate_grid_firing_maps(
    field_centers, peak_rates, peak_rate_probs, min_rate, diameter, gap, arena_map, rnd
):
    firing_rates = {}
    arena_min, arena_max = np.min(arena_map), np.max(arena_map)
    for gid in list(field_centers.keys()):
        max_rate = rnd.choice(peak_rates, p=peak_rate_probs)
        current_center = field_centers[gid]
        current_firing_rate = np.asarray(
            place_field_fr(current_center, arena_map, max_rate, diameter),
            dtype="float32",
        )
        current_pos = current_center - gap
        while current_pos >= arena_min:
            hopped_fr = np.asarray(
                place_field_fr(current_pos, arena_map, max_rate, diameter),
                dtype="float32",
            )
            current_firing_rate += hopped_fr
            current_pos -= gap
        current_pos = current_center + gap
        while current_pos <= arena_max:
            hopped_fr = np.asarray(
                place_field_fr(current_pos, arena_map, max_rate, diameter),
                dtype="float32",
            )
            current_firing_rate += hopped_fr
            current_pos += gap
        current_firing_rate[current_firing_rate <= min_rate] = min_rate
        firing_rates[gid] = current_firing_rate
    return firing_rates
