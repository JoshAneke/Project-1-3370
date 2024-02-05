[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/CfurQbKX)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11815509&assignment_repo_type=AssignmentRepo)
# COSC 3337 - Data Science I 
## Exploratory Data Analysis Assignment ##

### Due Date: September. 24, 11:59 PM ###

#### The goal of this assignment is to:
1. Learn how to pre-process datasets using statistics and visualizations
2. Learn how to interpret computed statistics and visualization to gain an understanding of the dataset 
3. Learn properties of the dataset and its characteristics that may lend to understanding of expected results from data analysis 
####

#### Dataset - mtcars
**Intended Outcome:**
The objective of Assignment 1 is to explore the relationships between different car attributes and their potential impact on fuel efficiency (mpg). Students will gain practical experience in data visualization, exploration, and statistical analysis.

#### Dataset: mtcars Data Set
The data was extracted from the 1974 Motor Trend US magazine, 
and includes information on fuel consumption and 10 aspects of 
automobile design and performance for 32 automobiles (1973–74 models).

**Attributes:**

- **model**: Car model name
- **mpg**: Miles/(US) gallon
- **cyl**: Number of cylinders
- **disp**: Displacement (cu.in.)
- **hp**: Gross horsepower
- **drat**: Rear axle ratio
- **wt**: Weight (1,000 lbs)
- **qsec**: 1/4 mile time
- **vs**: Engine shape (0 = V-shaped, 1 = straight)
- **am**: Transmission (0 = automatic, 1 = manual)
- **gear**: Number of forward gears
- **carb**: Number of carburetors

This dataset is a (11+1)D dataset and has been 
modified to include an additional nominal attribute, 
*efficiency*, as a class label to indicate fuel 
efficiency based on the value of the *mpg* attribute.
- *Class labels for 'efficiency':*
    - Low: mpg <= 15
    - Medium: 15 < mpg <= 25
    - High: mpg > 25

*Dataset modified by Tuck, Bryan, E.*

#### Assignment Tasks ####

1. Compute the covariance matrix for each pair of the following attributes:
    - disp
    - gear
    - wt
    - drat
    - cyl
    - mpg
2. Compute the correlations for each of the same pair of attributes.
3. Interpret the statistical findings based on the above two.
4. Create scatter plot for the attributes and interpret the plots:
    - disp/hp
    - wt/mpg
5. Create histograms for *wt*, *drat*, and *hp* attributes for the 3 *efficiency* targets (Low, Medium, and High), and interpret the resulting histograms.
6. Create box plots for *wt* and *disp* attributes for the instances of the 3 *efficiency* class (Low, Medium, and High), and a box plot for all instances in the dataset.  Interpret the resulting box plots.
7. Create supervised scatter plots for the following 3 pairs of attributes using the *efficiency* as a class variable: *wt* / *drat*, *hp* / *wt*, and *drat* / *hp*. Use different colors for the class variable. Interpret the obtained plots and address what can be said about the difficulty in predicting the *efficiency* and the distribution of the instances of the three classes.
8. Identify the best pair of attributes based on the generated supervised scatter plots to manually create a decision tree model to predict the *efficiency* class.  Explain how you built the decision tree and what you learned about the importance of the chosen attributes for the classification. 
9. Create a new dataset by transforming the *disp*, *hp*, *drat*, *wt*, and *qsec* attributes into z-scores. Fit a linear model that predicts the values of *mpg* attribute using the 5 z-scored continuous attributes as the independent variables. Report the R2 of the linear model and the coefficients of each attribute in the obtained regression function. What do the obtained coefficients tell you about the importance of each attribute for predicting the efficiency of a car?
10. Write a brief conclusion summarizing the most important findings of this task; in particular, address the findings obtained related to predicting the efficiency of a car. If possible, write about which attributes seem useful for predicting car efficiency and what you as an individual can learn from this dataset!

**Note:**

You are provided a shell python file with the tasks above in which you are to
develop your code.  Basic required python packages you will need are also specified
at start of the file via import statements.  

**Submission Instructions:**

Please push and commit your developed solution to the github repository.  Please ensure
that your Github username, name, and PSID are filled in at the start of the file.
Please ensure that all required plots are generated while running your solution and that
your interpretations and conclusions, as required for each task, are included as comments
after each task listed in the shell python file.

-----------------------

<sub><sup>
License: Property of Quantitative Imaging Laboratory (QIL), Department of Computer Science, University of Houston. This software is property of the QIL, and should not be distributed, reproduced, or shared online, without the permission of the author This software is intended to be used by students of the Data Programming course offered at University of Houston. The contents are not to be reproduced and shared with anyone with out the permission of the author. The contents are not to be posted on any online public hosting websites without the permission of the author. The software is cloned and is available to the students for the duration of the course. At the end of the semester, the Github organization is reset and hence all the existing repositories are reset/deleted, to accommodate the next batch of students.
</sub></sup>
