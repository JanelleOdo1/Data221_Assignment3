import pandas as pd

data_frame = pd.read_csv('crime.csv')
crime_col = data_frame['ViolentCrimesPerPop']

print("Mean:", crime_col.mean())
print("Median:", crime_col.median())
print("Standard Deviation:", crime_col.std())
print("Minimum:", crime_col.min())
print("Maximum:", crime_col.max())

# My answers
# Looking at the mean and median, if they are different, the data is skewed.
# Usually, crime data is right-skewed because most places have low crime but a few have very high crime.
# The mean is affected more by extreme values (outliers) because it adds everything up.
# The median is more stable since it's just the middle spot in the list.