import os

from livn.types import Model


class HippocampalDissociatedCulture(Model):

    def neuron_template_directory(self):
        return os.path.join(os.path.dirname(__file__), "templates")

    def neuron_mechanisms_directory(self):
        return os.path.join(os.path.dirname(__file__), "mechanisms")

    def neuron_celltypes(self, celltypes):
        for population, template_class in {
            "PYR": "templates.PyramidalCellBilash.PyramidalCell",
            "PVBC": "templates.PRN_neuron.PRN",
            "CCKBC": "templates.PRN_neuron.PRN",
            "AAC": "templates.PRN_neuron.PRN",
            "BS": "templates.PRN_neuron.PRN",
            "IS1": "templates.PR_neuron.PR",
            "IS2": "templates.PR_neuron.PR",
            "IS3": "templates.PR_neuron.PR",
            "IVY": "templates.PRN_neuron.PRN",
            "NGFC": "templates.PR_neuron.PR",
            "OLM": "templates.PRN_neuron.PRN",
            "SCA": "templates.PRN_neuron.PRN",
        }.items():
            celltypes[population][
                "template class"
            ] = f"benchmarks.hippocampal_dissociated_culture.{template_class}"

        return

    def neuron_synapse_mechanisms(self):
        return {
            "AMPA": "LinExp2Syn",
            "NMDA": "LinExp2SynNMDA",
            "GABA_A": "LinExp2Syn",
            "GABA_B": "LinExp2Syn",
        }

    def neuron_synapse_rules(self):
        return {
            "Exp2Syn": {
                "mech_file": "exp2syn.mod",
                "mech_params": ["tau1", "tau2", "e"],
                "netcon_params": {"weight": 0},
                "netcon_state": {},
            },
            "LinExp2Syn": {
                "mech_file": "lin_exp2syn.mod",
                "mech_params": ["tau_rise", "tau_decay", "e"],
                "netcon_params": {"weight": 0, "g_unit": 1},
                "netcon_state": {},
            },
            "LinExp2SynNMDA": {
                "mech_file": "lin_exp2synNMDA.mod",
                "mech_params": [
                    "tau_rise",
                    "tau_decay",
                    "e",
                    "mg",
                    "Kd",
                    "gamma",
                    "vshift",
                ],
                "netcon_params": {"weight": 0, "g_unit": 1},
                "netcon_state": {},
            },
            "SatExp2Syn": {
                "mech_file": "sat_exp2syn.mod",
                "mech_params": [
                    "sat",
                    "dur_onset",
                    "tau_offset",
                    "e",
                ],
                "netcon_params": {"weight": 0, "g_unit": 1},
                "netcon_state": {"onset": 2, "count": 3, "g0": 4, "t0": 5},
            },
        }

    def neuron_noise_mechanism(self, section):
        return None, None

    def neuron_noise_configure(
        self, population, mechanism, state, exc_level, inh_level
    ):
        pass

    def neuron_default_noise(self, system: str, key: int = 0):
        return {"exc": 0.0, "inh": 0.0}

    def neuron_default_weights(self, system: str):
        return {}
