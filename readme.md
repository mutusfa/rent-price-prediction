# Model for inferring monthly rent prices in Lithuania

## Introduction
I'm interested in invested in Real Estate in Lithuania and this project collects part of
the data needed to estimate viability of the venture.

I have scraped monthly rent prices for flats from a site for real estate listings
aruodas.lt, I will also need to collect data on real estate prices, that would provide a
ballpark estimate if buying real estate for renting it out is profitable and in what
regions it would be profitable.

## Workflow from A to Z

### Gathering data


```python
# !pip install git+https://github.com/mutusfa/scrape_aruodas
```


```python
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
```


```python
import pandas as pd
from pathlib import Path

from scrape_aruodas.main import scrape

PROJECT_DIR = Path.cwd()

%load_ext lab_black
```


```python
# raw = scrape(num_items=2000)
# raw.to_csv(str(PROJECT_DIR / "data/raw/rent.csv"))
```

### Cleaning the data


```python
import src.data.make_dataset

intermediate = (
    src.data.make_dataset.make_intermediate()
)  # also gets saved at data/intermediate/rent.csv
```

There is a notebook in src/data/select_final.ipynb with minimal exploratory data analysis, where I cut off outliers.


```python
final = pd.read_csv(str(PROJECT_DIR / "data/final/rent.csv"), index_col=0)
final.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>district</th>
      <th>latitude</th>
      <th>listing_url</th>
      <th>longitude</th>
      <th>street</th>
      <th>floor_area_m2</th>
      <th>monthly_rent</th>
      <th>number_of_rooms</th>
      <th>floor</th>
      <th>number_of_floors</th>
      <th>build_year</th>
      <th>building_type</th>
      <th>heating_type</th>
      <th>equipment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>vilniuje</td>
      <td>snipiskese</td>
      <td>54.720888</td>
      <td>https://www.aruodas.lt/butu-nuoma-vilniuje-sni...</td>
      <td>25.278539</td>
      <td>juozo-balcikonio-g</td>
      <td>19.0</td>
      <td>326.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>2020.0</td>
      <td>Mūrinis</td>
      <td>Centrinis kolektorinis</td>
      <td>Įrengtas</td>
    </tr>
    <tr>
      <th>2</th>
      <td>vilniuje</td>
      <td>naujininkuose</td>
      <td>54.662883</td>
      <td>https://www.aruodas.lt/butu-nuoma-vilniuje-nau...</td>
      <td>25.277840</td>
      <td>telsiu-g</td>
      <td>42.0</td>
      <td>399.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>2015.0</td>
      <td>Mūrinis</td>
      <td>Geoterminis</td>
      <td>Įrengtas</td>
    </tr>
    <tr>
      <th>3</th>
      <td>vilniuje</td>
      <td>fabijoniskese</td>
      <td>54.742411</td>
      <td>https://www.aruodas.lt/butu-nuoma-vilniuje-fab...</td>
      <td>25.229110</td>
      <td>salomejos-neries-g</td>
      <td>50.0</td>
      <td>360.0</td>
      <td>2.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>2008.0</td>
      <td>Mūrinis</td>
      <td>Kita</td>
      <td>Įrengtas</td>
    </tr>
    <tr>
      <th>5</th>
      <td>vilniuje</td>
      <td>senamiestyje</td>
      <td>54.681746</td>
      <td>https://www.aruodas.lt/butu-nuoma-vilniuje-sen...</td>
      <td>25.279369</td>
      <td>klaipedos-g</td>
      <td>105.0</td>
      <td>1500.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2013.0</td>
      <td>Mūrinis</td>
      <td>Centrinis kolektorinis</td>
      <td>Įrengtas</td>
    </tr>
    <tr>
      <th>6</th>
      <td>vilniuje</td>
      <td>naujininkuose</td>
      <td>54.662866</td>
      <td>https://www.aruodas.lt/butu-nuoma-vilniuje-nau...</td>
      <td>25.277922</td>
      <td>telsiu-g</td>
      <td>42.0</td>
      <td>350.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>2015.0</td>
      <td>Mūrinis</td>
      <td>Geoterminis</td>
      <td>Įrengtas</td>
    </tr>
  </tbody>
</table>
</div>



### Modelling


```python
import src.models

final = final.drop("listing_url", axis="columns")

model, scaler, encoder = src.models.main(final)
```

    Score: 0.668780779673825
    Mean absolute error: 128.82831971188514
    Mean absolute percentage error: 0.295953694179332


Running src.models as a script saves model and used scaler and encoder to ./models directory.

### Accessing model via api

#### Access to model's inference (post interface)

https://jjuoda-ds-24.herokuapp.com/predict/

For example, using requests library:


```python
import numpy as np
import requests

features_for_inference = [
    {
        "city": "kaune",
        "district": "centre",
        "latitude": 54.889328,
        "longitude": 23.936227,
        "street": "tunelio-g",
        "floor_area_m2": 25.0,
        "number_of_rooms": 1.0,
        "floor": 1.0,
        "number_of_floors": 2.0,
        "build_year": 1939.0,
        "building_type": "Medinis",
        "heating_type": "Dujinis",
        "equipment": "Įrengtas",
    }
]
url = "https://jjuoda-ds-24.herokuapp.com/predict/"
response = requests.post(url, json=features_for_inference)
inferred = np.array(response.json())
inferred
```




    array([198.87162398])



#### Access last inferences made

https://jjuoda-ds-24.herokuapp.com/inferences


```python
url = "https://jjuoda-ds-24.herokuapp.com/inferences/"
response = requests.get(url)
last_inferences = np.array(response.json())
last_inferences
```




    array([{'number_of_floors': 2.0, 'equipment': 'Įrengtas', 'number_of_rooms': 1.0, 'street': 'tunelio-g', 'longitude': 23.936227, 'district': 'centre', 'inferred_monthly_rent': 198.87162398292858, 'heating_type': 'Dujinis', 'build_year': 1939.0, 'floor': 1.0, 'floor_area_m2': 25.0, 'latitude': 54.889328, 'city': 'kaune', 'id': 85, 'building_type': 'Medinis'},
           {'number_of_floors': 2.0, 'equipment': 'Įrengtas', 'number_of_rooms': 1.0, 'street': 'tunelio-g', 'longitude': 23.936227, 'district': 'centre', 'inferred_monthly_rent': 198.87162398292858, 'heating_type': 'Dujinis', 'build_year': 1939.0, 'floor': 1.0, 'floor_area_m2': 25.0, 'latitude': 54.889328, 'city': 'kaune', 'id': 84, 'building_type': 'Medinis'},
           {'number_of_floors': 2.0, 'equipment': 'Įrengtas', 'number_of_rooms': 1.0, 'street': 'tunelio-g', 'longitude': 23.936227, 'district': 'centre', 'inferred_monthly_rent': 198.87162398292858, 'heating_type': 'Dujinis', 'build_year': 1939.0, 'floor': 1.0, 'floor_area_m2': 25.0, 'latitude': 54.889328, 'city': 'kaune', 'id': 83, 'building_type': 'Medinis'},
           {'number_of_floors': 2.0, 'equipment': 'Įrengtas', 'number_of_rooms': 1.0, 'street': 'tunelio-g', 'longitude': 23.936227, 'district': 'centre', 'inferred_monthly_rent': 198.87162398292858, 'heating_type': 'Dujinis', 'build_year': 1939.0, 'floor': 1.0, 'floor_area_m2': 25.0, 'latitude': 54.889328, 'city': 'kaune', 'id': 82, 'building_type': 'Medinis'},
           {'number_of_floors': 2.0, 'equipment': 'Įrengtas', 'number_of_rooms': 1.0, 'street': 'tunelio-g', 'longitude': 23.936227, 'district': 'centre', 'inferred_monthly_rent': 198.87162398292858, 'heating_type': 'Dujinis', 'build_year': 1939.0, 'floor': 1.0, 'floor_area_m2': 25.0, 'latitude': 54.889328, 'city': 'kaune', 'id': 81, 'building_type': 'Medinis'},
           {'number_of_floors': 2.0, 'equipment': 'Įrengtas', 'number_of_rooms': 1.0, 'street': 'tunelio-g', 'longitude': 23.936227, 'district': 'centre', 'inferred_monthly_rent': 198.87162398292858, 'heating_type': 'Dujinis', 'build_year': 1939.0, 'floor': 1.0, 'floor_area_m2': 25.0, 'latitude': 54.889328, 'city': 'kaune', 'id': 80, 'building_type': 'Medinis'},
           {'number_of_floors': 2.0, 'equipment': 'Įrengtas', 'number_of_rooms': 1.0, 'street': 'tunelio-g', 'longitude': 23.936227, 'district': 'centre', 'inferred_monthly_rent': 198.87162398292858, 'heating_type': 'Dujinis', 'build_year': 1939.0, 'floor': 1.0, 'floor_area_m2': 25.0, 'latitude': 54.889328, 'city': 'kaune', 'id': 79, 'building_type': 'Medinis'},
           {'number_of_floors': 2.0, 'equipment': 'Įrengtas', 'number_of_rooms': 1.0, 'street': 'tunelio-g', 'longitude': 23.936227, 'district': 'centre', 'inferred_monthly_rent': 198.8716239829286, 'heating_type': 'Dujinis', 'build_year': 1939.0, 'floor': 1.0, 'floor_area_m2': 25.0, 'latitude': 54.889328, 'city': 'kaune', 'id': 78, 'building_type': 'Medinis'},
           {'number_of_floors': 2.0, 'equipment': 'Įrengtas', 'number_of_rooms': 1.0, 'street': 'tunelio-g', 'longitude': 23.936227, 'district': 'centre', 'inferred_monthly_rent': 198.8716239829286, 'heating_type': 'Dujinis', 'build_year': 1939.0, 'floor': 1.0, 'floor_area_m2': 25.0, 'latitude': 54.889328, 'city': 'kaune', 'id': 77, 'building_type': 'Medinis'},
           {'number_of_floors': 2.0, 'equipment': 'Įrengtas', 'number_of_rooms': 1.0, 'street': 'tunelio-g', 'longitude': 23.936227, 'district': 'centre', 'inferred_monthly_rent': 198.87162398292858, 'heating_type': 'Dujinis', 'build_year': 1939.0, 'floor': 1.0, 'floor_area_m2': 25.0, 'latitude': 54.889328, 'city': 'kaune', 'id': 76, 'building_type': 'Medinis'}],
          dtype=object)