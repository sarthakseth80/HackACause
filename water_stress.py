def calculate_wsi(rainfall_deviation, groundwater_level, population):
    rainfall_score = min(rainfall_deviation, 50)
    groundwater_score = 50 - groundwater_level
    population_score = population / 1000

    wsi = (rainfall_score * 0.4) + \
          (groundwater_score * 0.3) + \
          (population_score * 0.3)

    return round(wsi, 2)


def classify_risk(wsi):
    if wsi > 70:
        return "Severe"
    elif wsi > 40:
        return "Moderate"
    else:
        return "Safe"