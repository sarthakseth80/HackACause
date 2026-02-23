def calculate_rainfall_deviation(normal_rainfall, current_rainfall):
    if normal_rainfall == 0:
        return 0

    deviation = ((normal_rainfall - current_rainfall) / normal_rainfall) * 100
    return round(deviation, 2)