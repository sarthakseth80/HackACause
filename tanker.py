def calculate_tanker_requirement(wsi, population):
    if wsi > 70:
        return population // 1000
    elif wsi > 40:
        return population // 2000
    else:
        return 0