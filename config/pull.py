from machinable import get
import yaml

get("machinable.index", "./index").__enter__()

e = get("interface._ca1_sopt_baseline", {"selection": [None]})
experiments = e.components.filter(lambda x: not x.load_attribute("preflight", False))


for p in e.populations():
    leader = None
    source = None
    for solution in experiments.filter(lambda x: x.population() == p):
        best = solution.get_best()
        if leader is None:
            leader = best
            source = solution
            continue

        if all(
            candidate < current
            for candidate, current in zip(
                best["y"].iloc[-1].to_list(), leader["y"].iloc[-1].to_list()
            )
        ):
            leader = best
            source = solution

    print(source.label())
    print(best["y"].iloc[-1].to_dict())
    print("-")

    x = best["x"].iloc[-1].to_dict()
    x.update(source.config.dopt_params.problem_parameters)
    print(x)

    DIR = "benchmarks/hippocampal_dissociated_culture/config"
    with open(f"{DIR}/CA1_{p}_PR_config.yaml") as f:
        default = yaml.load(f, yaml.SafeLoader)

        for k, v in x.items():
            if k not in ["dend_aqs_KAHP", "dend_bq_KAHP", "dend_gmax_KAHP"]:
                assert k in default["PinskyRinzel"], k
            default["PinskyRinzel"][k] = v

    with open(f"{DIR}/optimized/CA1_{p}_PR_config.yaml", "w") as f:
        yaml.dump(default, f)
