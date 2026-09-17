from datasets import load_dataset


def get_oscd(path):
    ds = load_dataset("ericyu/OSCD_Cropped_256", cache_dir=path)
    return ds


def get_levircd(path):
    ds = load_dataset("ericyu/LEVIRCD_Cropped256", cache_dir=path)
    return ds


def get_minenetcd(path):
    ds = load_dataset("HZDR-FWGEL/MineNetCD256", cache_dir=path)
    return ds


def get_sysu(path):
    ds = load_dataset("ericyu/SYSU_CD", cache_dir=path)
    return ds


def get_egybcd(path):
    ds = load_dataset("ericyu/EGY_BCD", cache_dir=path)
    return ds


def get_clcd(path):
    ds = load_dataset("ericyu/CLCD_Cropped_256", cache_dir=path)
    return ds


def get_gvlm(path):
    ds = load_dataset("ericyu/GVLM_Cropped_256", cache_dir=path)
    return ds


def main():
    p = "../datasets/hf"
    get_oscd(p)
    get_levircd(p)
    get_minenetcd(p)
    get_sysu(p)
    get_egybcd(p)
    get_clcd(p)
    get_gvlm(p)


if __name__ == "__main__":
    main()
