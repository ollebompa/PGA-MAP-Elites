from sklearn.model_selection import ParameterGrid

'''
File to generate experimental configs. The example makes it clear how it works,
'''

def get_base_conf(base_config):
    with open(base_config, "r") as base:
        base_dict = {}
        lines = base.readlines()
        for line in lines:
            line = line.strip("--").strip("\n").split("=")
            if len(line) == 1:
                base_dict[line[0]] = True
            else:
                base_dict[line[0]] = line[1]
        return base_dict


def write_conf_file(base_config, config, filename):
    with open(f"{filename}.txt", "w") as conf_file:
        lines = []
        for arg, val in base_config.items():
            if arg in config:
                val = config[arg]
                if isinstance(val, bool):
                    if val:
                        line = "--" + arg + "\n"
                    else:
                        continue
                else:
                    line = "--" + "=".join([arg, str(val)]) + "\n"
                
            else:
                if isinstance(val, str):
                    line = "--" + "=".join([arg, str(val)]) + "\n"
                else:
                    line = "--" + arg + "\n"
            lines.append(line)

        for arg, val in config.items():
            if arg not in base_config:
                if val:
                        line = "--" + arg + "\n"
                else:
                    continue
                lines.append(line)
        conf_file.writelines(lines)


def generate_grid_configs(varables, ranges):
    param_grid = {}
    for idx, var in enumerate(varables):
        param_grid[var] = ranges[idx]
    return list(ParameterGrid(param_grid))




if __name__ == "__main__":
    base_config = "QDAnt_config.txt"

    variables_1 = ["env", "seed", "dim_map", "n_niches", "max_evals", "neurons_list"]
    ranges_1 = [["QDAntBulletEnv-v0"], range(10), [4], [1000], [1000000], ["128 128"]]
    param_grid_1 = generate_grid_configs(variables_1, ranges_1)

    variables_2 = ["env", "seed", "dim_map", "n_niches", "max_evals", "neurons_list"]
    ranges_2 = [["QDWalker2DBulletEnv-v0"], range(10), [2], [1000], [1000000], ["128 128"]]
    param_grid_2 = generate_grid_configs(variables_2, ranges_2)

    variables_3 = ["env", "seed", "dim_map", "n_niches", "max_evals", "neurons_list"]
    ranges_3 = [["QDHalfCheetahBulletEnv-v0"], range(10), [2], [1000], [1000000], ["128 128"]]
    param_grid_3 = generate_grid_configs(variables_3, ranges_3)

    variables_4 = ["env", "seed", "dim_map", "n_niches", "max_evals", "neurons_list"]
    ranges_4 = [["QDHopperBulletEnv-v0"], range(10), [1], [1000], [1000000], ["128 128"]]
    param_grid_4 = generate_grid_configs(variables_4, ranges_4)


    configs =  param_grid_1 + param_grid_2 + param_grid_3 + param_grid_4
    print(configs)
    print(len(configs))


    base = get_base_conf(base_config)
    for i, conf in enumerate(configs):
        name = f"config_{i+1 + }"
        write_conf_file(base, conf, name)

