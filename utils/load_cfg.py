import yaml


def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
        return cfg


if __name__ == "__main__":
    path = "cfg.yaml"
    cfg = load_yaml(path)
    print(cfg)
