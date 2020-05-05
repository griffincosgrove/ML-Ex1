# Griffin Cosgrove
''' The purpose of this program is to provide a beginner example into machine learning. The program also includes in depth
comments for clarity. The program begins with importing our data from my GitHub Repo which is a CSV.
Following that we do some light exploratory data analysis with our data frame and visualize some data that might turn out
to be interesting. A heatmap is an important data visualization for feature selection so I included that in this exercise.
Following that we select our features and wrangle our data to be in the correct form for our linear regression model.
The final steps are to split our data into a training and test set and then train the model on the training set and test the
model on our test data. The test data is not in the training data, thus the model's performance is judged on its accuracy on
the test set. The final prints are the information on our linear regression model.
'''

# We are importing the necessary packages
# All of these libraries are included in the Anaconda Distribution of Python 3
# If you already have Python 3 installed, you can PIP Install all of
# these libraries thru command prompt "pip install <library name>"

import sklearn
# SK Learn or commonly refered to as SciKit Learn is the library that 
# allows us to create the algorithm and train it.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# matplotlib is a library that allows us to create powerful data visualizations
# based on the programming language Matlab
# line below allows the graphs to be displayed inside the Jupyter Notebook
from matplotlib import pyplot as plt

# Pandas library allows for data analysis and the ability to bring in our 
# data set in csv form.
import pandas

# Numpy is a library that contains support for multidimensional arrays
# and math functions both related and not related to arrays.
import numpy as np

# Seaborn is another library for data visualizations
import seaborn as sns

#end of imports

# df is the common variable name for your data frame.
# using pandas.read_excel to read the file and create a reference variable, df, for the data.
# Note: You must include the absolute path to your data source. So you will have to replace my path with YOUR path.
df = pandas.read_csv("https://raw.githubusercontent.com/griffincosgrove/ML-Ex1/master/housing.csv")

# Now that we have our data we can begin to analyze the data.
# We do this because we must pick features for our model.
# We do not always have to open the data file.
# The first step, if you have not already done so in excel, is to find out how many observations and possible features we have.
# shape tells us how many rows and columns there are in our data set. 
# Think rows = instances or observations, 
# columns = possible features
# so concretely in this case we have 41 observations(houses) and 10 possible features.
# by most classifications this is a small dataset. Modern data sets can contain over 1 million observations.
print("Remember, the following line is read as there are 41 observations and 10 possible features in our data set.")
print(df.shape)

# describe tells us summary statistics about our data set.
# This is helpful in with respect to exploratory data analysis that must be done prior to feature selection.
print(df.describe().round(2))

# Here we are going to create a scatter plot to visualize the relationship between the number of days
# the property was listed on zillow and the price of the property. My hunch is that the properties in the upper quartile
# of days listed will generally have lower prices.
# just some light exploratory data analysis before a deeper look into features

plt.scatter(df['days'], df['price'], color='green')
plt.xlabel("Days Listed on Zillow")
plt.ylabel("Price ($)")
plt.title("Relationship Between Days Listed on Zillow and List Price")

# Here we are creating a heat matrix. A heat matrix is useful because it visualizes how 
# Each variable correlates with every other variable and most importantly the variable we want to predict.
# annot = True displays the values inside the cells.
# linewidths creates a border between the cells for visualization purposes.
# Generally, we are looking for how variables correlate with our desired output, and how possible features correlate with eachother.
correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True, linewidths = .5)

# in-depth feature selection is beyond the scope of this assignment, but more research can be done independently.
# For this assignment I have selected lot size, area, bedroom, bathroom as my features

# combine multiple feature vectors to one 2D array using numpy
# this is where we define what we are using as inputs (features)
# and we define what we want the output to be in this case it will be the price of the house
X = np.column_stack((df.lot_size,df.area,df.bedroom,df.bathroom)) 
y = df.price

# now that we have defined X we can visualize it.
# Capital X in machine learning usually refers to an entire matrix where as lower case x would refer to a feature vector.
# printing the first 5 rows(observations) and their features for visualization purposes.
print("This is all of the features for the first 5 observations.")
print(X[:5])

# split data for training and testing
# I split the data into 80% training and 20% testing. This is the rule of thumb. 
# But in deep neural networks with data sets with around 1 million observations the split can be pushed as far as 95%/5%
# i have the random state set to 9 here. this is arbitrary. It is standard practice
# to choose a random state instead of leaving it as None because this allows our research to be reproduced by others
# notice the uppercase X
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2 , random_state=9)

# .fit is what performs the optmization, in this case minimization of the cost function for us.
lr = LinearRegression().fit(X_train, y_train)

# printing out the coefficient and intercept of the linear regression function, theta0 and theta1 up to thetaN from previous example
# the topic above is expanded on in the ppt if you are still uneasy.
# rounded to 4 decimal points for simplification
print("lr.coefficents_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

#R square value tells us how accurate a fit our line of best fit is/ linear regression function
print("Training R Square: {:.2f}".format(lr.score(X_train, y_train)))
print("Test R Square: {:.2f}".format(lr.score(X_test, y_test)))

# What this means is that our linear regression model was mostly accurate in predicting the price of a new house
# fed into the model

