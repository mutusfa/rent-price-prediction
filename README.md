# Model for inferring monthly rent prices in Lithuania

## Introduction
I'm interested in invested in Real Estate in Lithuania and this project collects part of
the data needed to estimate viability of the venture.

I have scraped monthly rent prices for flats from a site for real estate listings
aruodas.lt, I will also need to collect data on real estate prices, that would provide a
ballpark estimate if buying real estate for renting it out is profitable and in what
regions it would be profitable.

## Accessing the model

A greeter to check site works.
```https://jjuoda-ds-24.herokuapp.com/
```
Access to model's inference (post interface)
``` https://jjuoda-ds-24.herokuapp.com/predict/
```
Check (auto-generated) documentation:
``` https://jjuoda-ds-24.herokuapp.com/docs
```

## Installing the package
```pip install git+https://github.com/TuringCollegeSubmissions/jjuoda-DS.2.4
```
You will have to find your own data and put it into `data/raw/rent.csv',
`data/intermediate/rent.csv` or `data/final/rent.csv` depending on how prepared your
dataset is for training.

## Quirks with the package
That I'd solve if I had time or this was a customer facing app.

- Cities and districts are cased: 'vilniuje', and not 'Vilnius'
- I treat categorical inputs case sensitively.
