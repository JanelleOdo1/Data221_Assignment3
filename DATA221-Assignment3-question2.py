import pandas as pd
import matplotlib.pyplot as plt

data_frame = pd.read_csv('crime.csv')
crime_col = data_frame['ViolentCrimesPerPop']

plt.figure()
plt.hist(crime_col, bins=20, color='blue')
plt.title('Distribution of Violent Crimes')
plt.xlabel('Crimes Per Pop')
plt.ylabel('Frequency')
plt.show()

plt.figure()
plt.boxplot(crime_col)
plt.title('Box Plot of Violent Crimes')
plt.xlabel('Violent Crimes Column')
plt.ylabel('Value')
plt.show()

# My answers
# The histogram shows that the data is spread mostly at the low end[cite: 42].
# This confirms the skewness I talked about in Question 1[cite: 42].
# The box plot shows the median is closer to the bottom of the box[cite: 43].
# I can also see many points above the whiskers, which means there are several outliers[cite: 44].
# Overall, the data has a lot of variation between different communities[cite: 45].