import os
import modal


def generate_wine(quality, fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                  chlorides,total_sulfur_dioxide, pH, sulphates,
                  alcohol, red_wine, white_wine, sulfur_dioxide_ratio):
    """
    Returns a single wine as a single row in a DataFrame
    """
    import pandas as pd
    import random

    type_red = random.choice(red_wine)
    type_white = not type_red

    df = pd.DataFrame({ "fixed_acidity": [random.uniform(fixed_acidity[0], fixed_acidity[1])],
                       "volatile_acidity": [random.uniform(volatile_acidity[0], volatile_acidity[1])],
                       "citric_acid": [random.uniform(citric_acid[0], citric_acid[1])],
                       "residual_sugar": [random.uniform(residual_sugar[0], residual_sugar[1])],
                       "chlorides": [random.uniform(chlorides[0], chlorides[1])],
                       "total_sulfur_dioxide": [random.uniform(total_sulfur_dioxide[0], total_sulfur_dioxide[1])],
                       "ph": [random.uniform(pH[0], pH[1])],
                       "sulphates": [random.uniform(sulphates[0], sulphates[1])],
                       "alcohol": [random.uniform(alcohol[0], alcohol[1])],
                       "type_red": type_red, 
                       "type_white": type_white,
                       "sulfur_dioxide_ratio": [random.uniform(sulfur_dioxide_ratio[0], sulfur_dioxide_ratio[1])]

                      })
    df['quality'] = quality

    return df


def get_random_wine():
    """
    Returns a DataFrame containing one random wine
    """
    import pandas as pd
    import random


    intervals = [[(4.2, 11.8), (0.17, 1.58), (0.0, 0.66), (0.7, 16.2), (0.022, 0.267), (9.0, 440.0), (2.87, 3.63), (0.28, 0.86), (8.0, 12.6), (False, True), (False, True), (0.045871559633027525, 0.7083333333333334)],
      [(4.6, 12.5), (0.11, 1.13), (0.0, 1.0), (0.7, 17.55), (0.013, 0.61), (7.0, 272.0), (2.74, 3.9), (0.25, 2.0), (8.4, 13.5),  (False, True), (False, True), (0.033707865168539325, 0.7083333333333334)], 
      [(4.5, 15.9), (0.1, 1.33), (0.0, 1.0), (0.6, 23.5), (0.009, 0.611), (6.0, 344.0), (2.79, 3.79), (0.27, 1.98), (8.0, 14.9),  (False, True), (False, True), (0.023622047244094488, 0.782608695652174)], 
      [(3.8, 14.3), (0.08, 1.04), (0.0, 1.66), (0.7, 65.8), (0.015, 0.415), (6.0, 294.0), (2.72, 4.01), (0.23, 1.95), (8.4, 14.0), (False, True), (False, True), (0.022727272727272728, 0.8571428571428571)], 
      [(4.2, 15.6), (0.08, 0.915), (0.0, 0.76), (0.9, 19.25), (0.012, 0.358), (7.0, 289.0), (2.84, 3.82), (0.22, 1.36), (8.6, 14.2), (False, True), (False, True), (0.05, 0.7428571428571429)], 
      [(3.9, 12.6), (0.12, 0.85), (0.03, 0.74), (0.8, 14.8), (0.014, 0.121), (12.0, 212.5), (2.88, 3.72), (0.25, 1.1), (8.5, 14.0), (False, True), (False, True), (0.07894736842105263, 0.7555555555555555)], 
      [(6.6, 9.1), (0.24, 0.36), (0.29, 0.49), (1.6, 10.6), (0.018, 0.035), (85.0, 139.0), (3.2, 3.41), (0.36, 0.61), (10.4, 12.9), (False, False), (True, True), (0.19424460431654678, 0.4789915966386555)]]

    offset = 3
    quality = random.randint(3,9)
    index_of_quality = quality - offset

    wine_df = generate_wine(quality, intervals[index_of_quality][0],intervals[index_of_quality][1],intervals[index_of_quality][2],
                            intervals[index_of_quality][3], intervals[index_of_quality][4], intervals[index_of_quality][5],
                              intervals[index_of_quality][6], intervals[index_of_quality][7], intervals[index_of_quality][8], intervals[index_of_quality][9],
                               intervals[index_of_quality][10],intervals[index_of_quality][11])

   
    print(f"Add Wine of quality: {quality}")
    print(wine_df)
    return wine_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_df = get_random_wine()

    wine_fg = fs.get_feature_group(name="wine",version=2)

    wine_fg.insert(wine_df)

if __name__ == "__main__":
    
    g()
