"""main.py: Starter file for assignment on Exploratory Data Analysis """

__author__ = "Shishir Shah"
__version__ = "1.0.0"
__copyright__ = "Copyright 2023 by Shishir Shah, Quantitative Imaging Laboratory (QIL), Department of Computer  \
                Science, University of Houston.  All rights reserved.  This software is property of the QIL, and  \
                should not be distributed, reproduced, or shared online, without the permission of the author."

import pandas as pd # Package to read data files and store columns as a dataframe
import matplotlib.pyplot as plt # Package to support plots
import numpy as np # Package to support data types
import seaborn as sns # Package to support heatmap plots
from scipy.stats import zscore # Needed to perform z-score normalization
from sklearn.linear_model import LinearRegression # Package to support regression fits
from sklearn import metrics # Needed to compute metrics for linear fit model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score



__author__ = "Josh Aneke"
__version__ = "1.1.0"
'''
Github Username: JoshAneke  
PSID: 1828214
'''
pause_value = False

''' Read data file mtcars.csv '''
data = pd.read_csv("mtcars.csv", header=None)
data.columns = ['model', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb', 'efficiency']
data = data.drop(0)
print(data.head(10))
if (pause_value):
    pause = input('Press enter to continue...')


''' Compute and print covariance of specific attributes '''
print('Covariance/n')
cov_mat = data[['disp', 'gear']].cov()
cov_mat2 = data[['wt', 'drat']].cov()
cov_mat3 = data[['cyl', 'mpg']].cov()
print(cov_mat)
print(cov_mat2)
print(cov_mat3)
print()


''' Compute and print correlation of specific attributes '''
print('Correlation')
cor_mat = data[['disp', 'gear']].corr()
cor_mat2 = data[['wt', 'drat']].corr()
cor_mat3 = data[['cyl', 'mpg']].corr()
print(cor_mat)
print(cor_mat2)
print(cor_mat3)

''' My interpretations of the data characteristics based on computed
covariance and correlation measures '''
print()
''' The characteristcs for the data was exactly how I thought it was going to go. I combined the disp and gear, the wt and drat, and the cyl and mpg, and for the covariance, the values were all negative. This meant that the two variables were not similar to each other
Similarily, the correlation for all pairs on variables were negative also. This meant that for each and every pair, it shows that if one variable went down, the other variable would increase, and vice versa '''

''' Create and display scatter plot for specific attributes. '''
print('\nScatterplots')
fig, axes = plt.subplots(1, 2, figsize = (10, 4))

axes[0].scatter(data['disp'], data['hp'], color='red', marker='o')
axes[0].set_title('disp vs hp')
axes[0].set_xlabel('Displacement')
axes[0].set_ylabel('Gross Horsepower')

# Create the second scatter subplot
axes[1].scatter(data['wt'], data['mpg'], color='blue', marker='x')
axes[1].set_title('wt vs mpg')
axes[1].set_xlabel('Weight')
axes[1].set_ylabel('Miles per Gallon')

plt.show()


'''My interpretations of the resulting plots. '''
print()
''' In both the scatter plots, we could see that the points were similarly plotted, It looked mostly linear with a couple outliers '''

''' Create and display histograms for specific attributes for the 3 'efficiency' 
targets (Low, Medium, and High). '''
print('\nHistograms')
low_efficiency = data[data['efficiency'] == 'Low']
medium_efficiency = data[data['efficiency'] == 'Medium']
high_efficiency = data[data['efficiency'] == 'High']

plt.figure(figsize=(12, 8))  # Adjust figsize as needed

plt.subplot(3, 3, 1)
plt.hist(low_efficiency['wt'], bins=10, color='blue', alpha=0.7)
plt.xlabel('Weight (wt)')
plt.ylabel('Frequency')
plt.title('Low Efficiency - wt')

plt.subplot(3, 3, 2)
plt.hist(medium_efficiency['wt'], bins=10, color='green', alpha=0.7)
plt.xlabel('Weight (wt)')
plt.ylabel('Frequency')
plt.title('Medium Efficiency - wt')

plt.subplot(3, 3, 3)
plt.hist(high_efficiency['wt'], bins=10, color='red', alpha=0.7)
plt.xlabel('Weight (wt)')
plt.ylabel('Frequency')
plt.title('High Efficiency - wt')

plt.subplot(3, 3, 4)
plt.hist(low_efficiency['drat'], bins=10, color='blue', alpha=0.7)
plt.xlabel('Rear Axle Ratio (drat)')
plt.ylabel('Frequency')
plt.title('Low Efficiency - drat')

plt.subplot(3, 3, 5)
plt.hist(medium_efficiency['drat'], bins=10, color='green', alpha=0.7)
plt.xlabel('Rear Axle Ratio (drat)')
plt.ylabel('Frequency')
plt.title('Medium Efficiency - drat')

plt.subplot(3, 3, 6)
plt.hist(high_efficiency['drat'], bins=10, color='red', alpha=0.7)
plt.xlabel('Rear Axle Ratio (drat)')
plt.ylabel('Frequency')
plt.title('High Efficiency - drat')

plt.subplot(3, 3, 7)
plt.hist(low_efficiency['hp'], bins=10, color='blue', alpha=0.7)
plt.xlabel('Horsepower (hp)')
plt.ylabel('Frequency')
plt.title('Low Efficiency - hp')

plt.subplot(3, 3, 8)
plt.hist(medium_efficiency['hp'], bins=10, color='green', alpha=0.7)
plt.xlabel('Horsepower (hp)')
plt.ylabel('Frequency')
plt.title('Medium Efficiency - hp')

plt.subplot(3, 3, 9)
plt.hist(high_efficiency['hp'], bins=10, color='red', alpha=0.7)
plt.xlabel('Horsepower (hp)')
plt.ylabel('Frequency')
plt.title('High Efficiency - hp')

plt.tight_layout()

plt.show()
''' My interpretations of the resulting plots. '''
''''My interpretation for the resulting plots is that there were way more medium effieciency than any other ones. Also, for the weight, the lower the weight, the higher the effienct. For the Rear Axle Ratio, the higher the drat, the higher the effieciency. Lastly, for the horsepower, the lower the hp, the higher the efficiency '''

'''Create and display box plots for specific attributes for the 3 'efficiency' 
targets (Low, Medium, and High). '''
print("\nBoxplot")
data['wt'] = pd.to_numeric(data['wt'], errors='coerce')  # 'coerce' to handle non-convertible values
data['mpg'] = pd.to_numeric(data['mpg'], errors='coerce')
data['cyl'] = pd.to_numeric(data['cyl'], errors='coerce')
data['disp'] = pd.to_numeric(data['disp'], errors='coerce')
data['hp'] = pd.to_numeric(data['hp'], errors='coerce')
data['drat'] = pd.to_numeric(data['drat'], errors='coerce')
data['qsec'] = pd.to_numeric(data['qsec'], errors='coerce')
data['vs'] = pd.to_numeric(data['vs'], errors='coerce')
data['am'] = pd.to_numeric(data['am'], errors='coerce')
data['gear'] = pd.to_numeric(data['gear'], errors='coerce')
data['carb'] = pd.to_numeric(data['disp'], errors='coerce')



plt.figure(figsize=(14, 6))  # Adjust figsize as needed

# Box plots for 'wt' and 'disp' attributes grouped by 'efficiency'
plt.subplot(1, 3, 1)
sns.boxplot(x='efficiency', y='wt', data=data, palette='Set1')
plt.xlabel('Efficiency Class')
plt.ylabel('Weight (wt)')
plt.title('Weight (wt) by Efficiency Class')

plt.subplot(1, 3, 2)
sns.boxplot(x='efficiency', y='disp', data=data, palette='Set2')
plt.xlabel('Efficiency Class')
plt.ylabel('Displacement (disp)')
plt.title('Displacement (disp) by Efficiency Class')

''' Create and display box plot for all instances of select attributes in the dataset. '''
plt.subplot(1, 3, 3)
sns.boxplot(data=data, palette='Set1')
plt.xlabel('Attributes')
plt.ylabel('Values')
plt.title('Box Plot of Weight (wt) and Displacement (disp) for All Instances')

# Adjust layout
plt.tight_layout()

# Show the box plots
plt.show()

''' My interpretations of the resulting plots. '''
''' The box plots suprisingly look similar to the histogram. The median for all tends to be closer to the middle except for the wt medium class. Overall, it looks like how I imagined it '''

''' Create and display supervised scatter plots for specified pairs of attributes 
using 'efficiency' as a class variable. '''
plt.figure(figsize=(12, 4))  # Adjust figsize as needed

# Scatter plot for wt/drat with color-coded classes
plt.subplot(1, 3, 1)
sns.scatterplot(x='wt', y='drat', hue='efficiency', data=data, palette='Set1')
plt.xlabel('Weight (wt)')
plt.ylabel('Drat')
plt.title('Scatter Plot: wt vs. drat')

# Scatter plot for hp/wt with color-coded classes
plt.subplot(1, 3, 2)
sns.scatterplot(x='hp', y='wt', hue='efficiency', data=data, palette='Set2')
plt.xlabel('Horsepower (hp)')
plt.ylabel('Weight (wt)')
plt.title('Scatter Plot: hp vs. wt')

# Scatter plot for drat/hp with color-coded classes
plt.subplot(1, 3, 3)
sns.scatterplot(x='drat', y='hp', hue='efficiency', data=data, palette='Set3')
plt.xlabel('Drat')
plt.ylabel('Horsepower (hp)')
plt.title('Scatter Plot: drat vs. hp')

# Adjust layout
plt.tight_layout()

# Show the scatter plots
plt.show()

''' My interpretations of the resulting plots. '''
''' 'My interpretations of the plot are also similar to the histograms, but this graph is much more clear than the others. You can clearly see the relation with the efficiencies between the medium, high, and low for the graph. For the wt vs drat graph, Its more of a linear graph going negative. On the hp vs wt, its more of a positive approach, and for the drat vs hp, the graph is very sporati and has many random outliers. This shows that it can be very difficult to predict the difficulty and distribution of the instances in the 3 classes'''

''' Best pair of attributes identified based on the generated supervised scatter 
plots and description of a decision tree model to predict the 'efficiency' class.  
My explanation of how I built the decision tree and what I learned about the importance 
of the chosen attributes for  classification. '''
print()
''''The best pair that you should choose based on the plots is the hp vs wt plot. This plot should be chosen becuase it has the clearest discrepencies between each efficiency level
You could build a decision tree with the base node being if the horsepower is above 2.2. This would seperate into a high efficiency and a root that would see if the horsepower was higher than 255. This tree would seperate into two branches with the left being medium and the right being low. 
The way I chose was a binary decision. This was because in the graph, there were clear distinctions between where each effiiciency ends and where each efficiency starts. Also, since all the values are continuous, I thought it would be more efficient. '''

''' Generating a new dataset by transforming specific attributes into z-scores. '''
numeric_attributes = ['disp', 'hp', 'drat', 'wt', 'qsec']

# Initialize the StandardScaler
scaler = StandardScaler()

# Standardize the numeric attributes
data[numeric_attributes] = scaler.fit_transform(data[numeric_attributes])

# Print the DataFrame with standardized numeric attributes
print(data[numeric_attributes])
''' Fitting a linear model that predicts the values of 'mpg' attribute using the 
computed z-scored continuous attributes as the independent variables. '''
X = data[numeric_attributes]
y = data['mpg']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the target values on the test set
y_pred = model.predict(X_test)
print(y_pred)

''' Compute and print the R2 of the linear model and the coefficients of each attribute. '''
# Calculate the R-squared (R2) as a measure of model performance
r2 = r2_score(y_test, y_pred)
coefficients = model.coef_
intercept = model.intercept_

# Print the R-squared (R2) and coefficients
print("R-squared (R2):", r2)
print("Coefficients (Slopes):", coefficients)
print("Intercept:", intercept)


''' What do the obtained coefficients tell me about the importance of each attribute
for predicting the efficiency of a car. '''
''' the obtained coefficients can help you prioritize attributes by their impact on car efficiency. Attributes with larger, statistically significant coefficients are likely to be more important predictors, while the attributes with smaller coefficients have less important predictors. Also positive coefficients suggest that an increase in the attributes value is associated with an increase in car efficiency, whereas Negative coefficients suggest that an increase in the attributes value is associated with a decrease in car efficiency '''

''' Brief conclusion summarizing the most important findings of this task. '''
''' There were many important findings in this. We learned how to make many types of graphs, such as scatter, supervised scatter, box, histogram, and decision trees. We learned the different parts of the linear model too. n conclusion, this analysis highlights the importance of specific attributes, particularly weight (wt) and horsepower (hp), in predicting car efficiency. These attributes showed clear distinctions between efficiency classes and played a significant role in the linear regression model. As an individual, you can learn that car efficiency is influenced by attributes like weight and horsepower, and these attributes can be reliable predictors for determining efficiency classes. This knowledge can be valuable for decision-making related to car design and optimization, contributing to more fuel-efficient and environmentally friendly vehicles. '''