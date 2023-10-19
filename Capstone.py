#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# # Ames Housing Project Suggestions
# 
# Data science is not a linear process. In this project, in particular, you will likely find that EDA, data cleaning, and exploratory visualizations will constantly feed back into each other. Here's an example:
# 
# 1. During basic EDA, you identify many missing values in a column/feature.
# 2. You consult the data dictionary and use domain knowledge to decide _what_ is meant by this missing feature.
# 3. You impute a reasonable value for the missing value.
# 4. You plot the distribution of your feature.
# 5. You realize what you imputed has negatively impacted your data quality.
# 6. You cycle back, re-load your clean data, re-think your approach, and find a better solution.
# 
# Then you move on to your next feature. _There are dozens of features in this dataset._
# 
# Figuring out programmatically concise and repeatable ways to clean and explore your data will save you a lot of time.

# The final task of this capstone project is to create a presentation based on the outcomes of all tasks in previous modules and labs.
# Your presentation will develop into a story of all your data science journey in this project, and it should be compelling and easy to understand.
# 
# In the next exercise, you can find a provided PowerPoint template to help you get started to create a report in slides format.
# However, you are free to add additional slides, charts, and tables.
# 
# Note that this presentation will be prepared for your peer-data-scientists whom are eager to understand every technical detail of this project.
# As such, this presentation will be much more detailed and technical than regular high-level and abstracted presentation for your executive team.
# 
# Once you have completed a detailed report, it should be straightforward for you to abstract it into a high-level
# deck for your executive team and/or stakeholders.

# In[2]:


import dash
import folium
import base64
import warnings
import numpy as np
import scipy.stats
import pandas as pd
import seaborn as sns
from io import BytesIO
import plotly.express as px
from dash import Input, Output
import matplotlib.pyplot as plt
from pandasql import sqldf as psql
import dash_core_components as dcc
import dash_html_components as html
from IPython.display import display
from geopy.geocoders import Nominatim
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

# Ignore all warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def read_csv(file_name):
    dataframe = pd.read_csv(f'{file_name}.csv')
    return dataframe

train_df = read_csv('train')
test_df = read_csv('test')


# ## Exploratory Data Analysis

# In[4]:


def data_overview(dataframe):
    # Get info on our columns and data size
    dataframe.info(memory_usage = 'deep')
    
    # Get Statistics for the Dataframe
    print(dataframe.describe())

def data_cleaning(dataframe):
    # Handle missing values
    for column in dataframe.columns:
        dataframe[column].replace(np.nan, 0, inplace = True)
    
    # Remove duplicates
    dataframe.drop_duplicates(inplace = True)
    
    return dataframe
    
def pairplot_visualization(dataframe):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Pair plot (for multiple variables)
    sns.pairplot(dataframe)
    plt.show()
    
def heatmap(dataframe):
    plt.figure(figsize = (20,20))
    sns.heatmap(np.round(dataframe.corr(), 2), annot = True, cmap = 'coolwarm')
    
def histogram(dataframe):
    # Plots subplots of all the features in the data frame.
    dataframe.hist(figsize = (15, 15));


# ### Train Dataframe Analysis

# In[5]:


train_df.head()


# In[6]:


data_overview(train_df)


# In[7]:


data_cleaning(train_df)


# In[8]:


pairplot_visualization(train_df)


# In[9]:


heatmap(train_df)


# In[10]:


histogram(train_df)


# #### SQL Analysis

# In[12]:


# Selecting the first 5 rows
query = "SELECT * FROM train_df LIMIT 5"
result = pd.DataFrame(psql(query, locals()))
print("Example 1 - First 5 rows:")
result


# In[13]:


# Example 2: Filtering rows with specific conditions
query = 'SELECT * FROM train_df WHERE SalePrice < 200000 AND "Yr Sold" = 2010'
result = pd.DataFrame(psql(query, locals()))
print("\nExample 2 - Rows with SalePrice < $200,000 and Yr Sold = 2010:")
result


# In[14]:


# Example 3: Aggregating data
query = 'SELECT Neighborhood, AVG(SalePrice) AS AvgSalePrice FROM train_df GROUP BY Neighborhood'
result = pd.DataFrame(psql(query, locals()))
print("\nExample 3 - Average SalePrice by Neighborhood:")
result


# In[15]:


# SQL query to calculate the minimum, maximum, average, and median SalePrice
sql_query = """
SELECT
    MIN(SalePrice) AS MinSalePrice,
    MAX(SalePrice) AS MaxSalePrice,
    AVG(SalePrice) AS AvgSalePrice,
    (SELECT SalePrice FROM train_df ORDER BY SalePrice LIMIT 1 OFFSET (SELECT COUNT(*) FROM train_df) / 2) AS MedianSalePrice
FROM
    train_df
"""

# Execute the SQL query using pandasql
result = pd.DataFrame(psql(sql_query, locals()))

# Display the result
print("How expensive are houses?")
print("The cheapest house sold for ${:,.0f} and the most expensive for ${:,.0f}".format(result['MinSalePrice'][0], result['MaxSalePrice'][0]))
print("The average sales price is ${:,.0f}, while median is ${:,.0f}".format(result['AvgSalePrice'][0], result['MedianSalePrice'][0]))


# #### Univariate Analysis

# In[16]:


def get_feature_groups(dataframe):
    """ Returns a list of numerical and categorical features,
    excluding SalePrice and Id. """
    # Numerical Features
    num_features = dataframe.select_dtypes(include = ['int64','float64']).columns
    num_features = num_features.drop(['Id','SalePrice']) # drop ID and SalePrice

    # Categorical Features
    cat_features = dataframe.select_dtypes(include = ['object']).columns
    return list(num_features), list(cat_features)

num_features, cat_features = get_feature_groups(train_df)


# In[17]:


# Let's start with our dependent variable, SalePrice
plt.figure(figsize = (10,6))
sns.distplot(train_df.SalePrice)
plt.show()


# We see that SalePrice is positively skewed. In fact, we get:

# In[18]:


print('Skew: {:.3f} | Kurtosis: {:.3f}'.format(
    train_df.SalePrice.skew(), train_df.SalePrice.kurtosis()))


# We see a bunch of features that look positively skewed, similar to SalePrice. We'll want to log transform
# these, include: LotFrontage, LotArea, BsmtUnfSF, TotalBsmtSF, 1stFlrSF, GrLivAre, GarageArea

# In[19]:


# Grid of distribution plots of all numerical features
f = pd.melt(train_df, value_vars = sorted(num_features))
g = sns.FacetGrid(f, col = 'variable', col_wrap = 4, sharex = False, sharey = False)
g = g.map(sns.distplot, 'value')


# MSSubclass should really be categorical, and we make a note to ourselves to take care of this when we process the data later on. For purposes of regression, we should also treat MoSold as categorical as the Euclidean distance between them doesn't make sense in this application. Same for YrSold. For YearBuilt, however, the distance is relevant as it implies age of the house.

# In[20]:


# Percentage of zero values
count_features = ['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
                  'KitchenAbvGr','TotalRmsAbvGr','Fireplaces','GarageCars']
non_count_features = [f for f in num_features if f not in count_features]
sparse_features = (train_df[non_count_features] == 0).sum() / train_df.shape[0]
sparse_features[sparse_features > 0].sort_values(ascending = True).plot(kind = 'barh', figsize = (10,6))
plt.title('Level of Sparsity')
plt.show()


# The categorical features will be much more interesting when compaired to our target feature SalePrice, but we can note a couple of things nevertheless. First, we note that there are plenty of feature were one value is heavily overrpresented, e.g. Condition2 (Proximity to various conditions (if more than one is present)), where nearly 99% of houses are listed as "Norm". That's fine though, as those edge cases may help us predict outliers. The second thing to realize is that a number of categorical features actually contain rank information in them and should thus be converted to discrete quantitative features similar to OverallQual.

# In[21]:


# First off, earlier we said we'll need to transform
# a couple features to categorical. Since we're looking 
# at categorical data here, let's go ahead and do that now
# so they are included in the analysis.
train_df['MS SubClass'] = train_df['MS SubClass'].apply(lambda x: str(x))
train_df['Mo Sold'] = train_df['Mo Sold'].apply(lambda x: str(x))
train_df['Yr Sold'] = train_df['Yr Sold'].apply(lambda x: str(x))

# Update our list of numerical and categorical features
num_features, cat_features = get_feature_groups(train_df)

# Count plots of categorical features
f = pd.melt(train_df, value_vars = sorted(cat_features))
g = sns.FacetGrid(f, col = 'variable', col_wrap = 4, sharex = False, sharey = False)
plt.xticks(rotation = 'vertical')
g = g.map(sns.countplot, 'value')
[plt.setp(ax.get_xticklabels(), rotation = 60) for ax in g.axes.flat]
g.fig.tight_layout()
plt.show()


# #### Bivariate Analysis

# In[22]:


# We want to change from categorical to numerical because the contain ranked information (e.g. quality ratings)
# We'll make those transforms now already so that they are properly included in the following section.
# We're also going to replace missing values with 0 already.

# Alley
train_df.Alley.replace({'Grvl':1, 'Pave':2}, inplace = True)
# Lot Shape
train_df['Lot Shape'].replace({'Reg':1, 'IR1':2, 'IR2':3, 'IR3':4}, inplace = True)
# Land Contour
train_df['Land Contour'].replace({'Low':1, 'HLS':2, 'Bnk':3, 'Lvl':4}, inplace = True)
# Utilities
train_df.Utilities.replace({'ELO':1, 'NoSeWa':2, 'NoSewr':3, 'AllPub':4}, inplace = True)
# Land Slope
train_df['Land Slope'].replace({'Sev':1, 'Mod':2, 'Gtl':3}, inplace = True)
# Exterior Quality
train_df['Exter Qual'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace = True)
# Exterior Condition
train_df['Exter Cond'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace = True)
# Basement Quality
train_df['Bsmt Qual'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace = True)
# Basement Condition
train_df['Bsmt Cond'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace = True)
# Basement Exposure
train_df['Bsmt Exposure'].replace({'No':1, 'Mn':2, 'Av':3, 'Gd':4}, inplace = True)
# Finished Basement 1 Rating
train_df['BsmtFin Type 1'].replace({'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}, inplace = True)
# Finished Basement 2 Rating
train_df['BsmtFin Type 2'].replace({'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}, inplace = True)
# Heating Quality and Condition
train_df['Heating QC'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace = True)
# Kitchen Quality
train_df['Kitchen Qual'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace = True)
# Home functionality
train_df.Functional.replace({'Sal':1, 'Sev':2, 'Maj2':3, 'Maj1':4, 'Mod':5, 'Min2':6, 'Min1':7, 'Typ':8}, inplace = True)
# Fireplace Quality
train_df['Fireplace Qu'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace = True)
# Garage Finish
train_df['Garage Finish'].replace({'Unf':1, 'RFn':2, 'Fin':3}, inplace = True)
# Garage Quality
train_df['Garage Qual'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace = True)
# Garage Condition
train_df['Garage Cond'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace = True)
# Paved Driveway
train_df['Paved Drive'].replace({'N':1, 'P':2, 'Y':3}, inplace = True)
# Pool Quality
train_df['Pool QC'].replace({'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace = True)

# We'll set all missing values in our newly converted features to 0
converted_features = ['Alley','Lot Shape', 'Land Contour', 'Utilities', 'Land Slope', 
                      'Exter Qual','Exter Cond', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure',
                      'BsmtFin Type 1','BsmtFin Type 2','Heating QC', 'Kitchen Qual', 'Functional',
                      'Fireplace Qu','Garage Finish','Garage Qual', 'Garage Cond','Paved Drive','Pool QC']

train_df[converted_features] = train_df[converted_features].fillna(0)

# Update our list of numerical and categorical features
num_features, cat_features = get_feature_groups(train_df)

# Scatter plots of numerical features against SalePrice
f = pd.melt(train_df, id_vars = ['SalePrice'], value_vars = sorted(num_features))
g = sns.FacetGrid(f, col = 'variable', col_wrap = 4, sharex = False, sharey = False)
plt.xticks(rotation = 'vertical')
g = g.map(sns.regplot, 'value', 'SalePrice', scatter_kws = {'alpha':0.3})
[plt.setp(ax.get_xticklabels(), rotation = 60) for ax in g.axes.flat]
g.fig.tight_layout()
plt.show()


# We find that there are quite a few features that seem to show strong correlation to SalePrice, such as OverallQual, TotalBsmtSF, GrLivArea, and TotRmsAbvGrd.

# In[23]:


plt.figure(figsize = (12,6))
plt.subplot(121)
sns.regplot(train_df['Gr Liv Area'], train_df.SalePrice, scatter_kws = {'alpha':0.3})
plt.title('Cone shape visible before log transform')

plt.subplot(122)
sns.regplot(np.log1p(train_df['Gr Liv Area']), np.log1p(train_df.SalePrice), scatter_kws = {'alpha':0.3})
plt.title('After log transform')
plt.show()


# In[24]:


# Feature sorted by correlation to SalePrice, from positive to negative
corr = train_df[['SalePrice'] + num_features].corr()
corr = corr.sort_values('SalePrice', ascending = False)
plt.figure(figsize = (8,10))
sns.barplot(corr.SalePrice[1:], corr.index[1:], orient = 'h')
plt.show()


# It looks like some features show significant variance in the mean of SalePrice between different groups, eg. Neighborhood, SaleType or MSSubClass.
# 
# However, we’d like to have a better sense of which feature influences SalePrice more than others. What we’ll do is run one-way ANOVA tests for each categorical feature againt SalePrice. This will give us both the F statistic and p-values for each feature. The higher the F statistic, the higher the p-value (i.e. the more confident we can be in rejecting the null hypothesis), but since the p-value will take into consideration a given F distribution (based on number of groups and number of observations), we will ultimately sort the features by p-value (instead of F). What does the p-value tell us? Again it tells how confident we can be in rejecting the null hypothesis; put differently, it answers the question "how likely was it to see data for each group if each group had in reality no effect on the dependent variable?" The unlikelier it is, the greater the difference in groups and therefore the more significant of an influence the feature has on the dependent variable SalePrice.

# In[25]:


# Count plots of categorical features
f = pd.melt(train_df, id_vars = ['SalePrice'], value_vars = sorted(cat_features))
g = sns.FacetGrid(f, col = 'variable', col_wrap = 3, sharex = False, sharey = False, size = 4)
g = g.map(sns.boxplot, 'value', 'SalePrice')
[plt.setp(ax.get_xticklabels(), rotation = 90) for ax in g.axes.flat]
g.fig.tight_layout()
plt.show()


# In[26]:


# In order for ANOVA to work, we have to take care of missing values first
train_df[cat_features] = train_df[cat_features].fillna('Missing')

# Onward...
anova = {'feature':[], 'f':[], 'p':[]}
for cat in cat_features:
    group_prices = []
    for group in train_df[cat].unique():
        group_prices.append(train_df[train_df[cat] == group]['SalePrice'].values)
    f, p = scipy.stats.f_oneway(*group_prices)
    anova['feature'].append(cat)
    anova['f'].append(f)
    anova['p'].append(p)
anova = pd.DataFrame(anova)
anova = anova[['feature','f','p']]
anova.sort_values('p', inplace = True)

# Plot
plt.figure(figsize = (14,6))
sns.barplot(anova.feature, np.log(1./anova['p']))
plt.xticks(rotation = 90)
plt.show()


# Of all our categorical features, Neighborhood appears to have the greatest influence on SalePrice. It's important to note here that the chart really undersells how much more influence Neighborhood has. We took the log of the inverse of the p-value (np.log(1./anova['p']): the inverse so that when we take the log we get positive numbers with the p-value having a magnitude of about 300x smaller than the next feature.

# In[27]:


# First we visually inspect a scatter plot of GrLivArea vs. SalePrice
plt.figure(figsize = (10,6))
sns.regplot(train_df['Gr Liv Area'], train_df.SalePrice, scatter_kws = {'alpha':0.3})
plt.show()


# ### Creating a Folium Map
# 
# Need to obtain the coordinates of all my neighborhoods in Ames, Iowa using Pandas and GeoPy. After obtaining the coordinates, it creates a map.

# In[28]:


# Create a geocoder (Nominatim is a free geocoding service)
geolocator = Nominatim(user_agent = "neighborhood_geocoder")

train_df_copy = train_df.copy()

# Function to get coordinates for a location
def get_coordinates(location):
    try:
        location = geolocator.geocode(location, timeout = 10)  # Increase the timeout if needed
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        print(f"Error geocoding {location}: {str(e)}")
        return None, None

# Calculate coordinates and add new columns
train_df_copy[['Latitude', 'Longitude']] = train_df_copy['Neighborhood'].apply(get_coordinates).apply(pd.Series)

# Create a map centered at an initial location
m = folium.Map(location = [42.0308, -93.6319], zoom_start = 13)

# Iterate through each row in the DataFrame
for index, row in train_df_copy.iterrows():
    # Check if the Latitude and Longitude columns have valid values
    if not pd.isna(row['Latitude']) and not pd.isna(row['Longitude']):
        # Add a marker for each coordinate
        folium.Marker(
            location = [row['Latitude'], row['Longitude']],
            tooltip = f"Neighborhood: {row['Neighborhood']}",
        ).add_to(m)

# Display the map in the notebook
display(m)


# #### Feature Engineering

# In[29]:


# Total Square Footage
train_df['TotalSF'] = train_df['Total Bsmt SF'] + train_df['Gr Liv Area']
train_df['TotalFloorSF'] = train_df['1st Flr SF'] + train_df['2nd Flr SF']
train_df['TotalPorchSF'] = train_df['Open Porch SF'] + train_df['Enclosed Porch'] + train_df['3Ssn Porch'] + train_df['Screen Porch']
    
# Total Bathrooms
train_df['TotalBathrooms'] = train_df['Full Bath'] + .5 * train_df['Half Bath'] + train_df['Bsmt Full Bath'] + .5 * train_df['Bsmt Half Bath']

# Booleans
train_df['HasBasement'] = train_df['Total Bsmt SF'].apply(lambda x: 1 if x > 0 else 0)
train_df['HasGarage'] = train_df['Garage Area'].apply(lambda x: 1 if x > 0 else 0)
train_df['HasPorch'] = train_df.TotalPorchSF.apply(lambda x: 1 if x > 0 else 0)
train_df['HasPool'] = train_df['Pool Area'].apply(lambda x: 1 if x > 0 else 0)
train_df['WasRemodeled'] = (train_df['Year Remod/Add'] != train_df['Year Built']).astype(np.int64)
train_df['IsNew'] = (train_df['Year Built'] > 2000).astype(np.int64)

boolean_features = ['HasBasement', 'HasGarage', 'HasPorch', 'HasPool', 
                    'WasRemodeled', 'IsNew']

num_features, cat_features = get_feature_groups(train_df)
num_features = [f for f in num_features if f not in boolean_features]


# #### Transforms
# Some of the numerical features exhibit positive skew and could benefit from a log transform. Let's go ahead and do that now.

# In[30]:


# Here we will be simplistic about it and simply
# log transform any numerical feature with a 
# skew greater than 0.5
features = num_features + ['SalePrice']
for f in features:
    train_df.loc[:,f] = np.log1p(train_df[f])


# #### One-Hot Encoding

# In[31]:


# before we continue, let's drop some cols
y = train_df['SalePrice']
train_df.drop('SalePrice', axis = 1, inplace = True)
train_df.drop('Id', axis = 1, inplace = True)


# In[32]:


model_data = pd.get_dummies(train_df).copy()


# ### Test Dataframe Analysis

# In[33]:


test_df.head()


# In[34]:


data_overview(test_df)


# In[35]:


data_cleaning(test_df)


# In[36]:


pairplot_visualization(test_df)


# In[37]:


heatmap(test_df)


# In[38]:


histogram(test_df)


# ## Modeling

# In[39]:


# Split data intro train and validation sets
X_train, X_test, y_train, y_test = train_test_split(model_data.copy(), y, test_size = 0.3, random_state = 42)
print('Shapes')
print('X_train:', X_train.shape)
print('X_val:', X_test.shape)
print('y_train:', y_train.shape)
print('y_val:', y_test.shape)


# #### Standardization

# In[40]:


# We'll use the convenient sklearn RobustScaler.
# Note we're only standardizing numerical features, not
# the dummy features. The RobustScaler helps us deal with outliers.
stdsc = StandardScaler()
X_train.loc[:,num_features] = stdsc.fit_transform(X_train[num_features])
X_test.loc[:,num_features] = stdsc.transform(X_test[num_features])


# #### Error Handling

# In[41]:


def rsme(model, X, y):
    cv_scores = -cross_val_score(model, X, y, scoring = 'neg_mean_squared_error', cv = 10)
    return np.sqrt(cv_scores)


# ### Ordinary Least Squares (OLS) Linear Regression, No Regularization
# 
# Start by doing a simple OLS LR and see how we do. Linear regression models can be heavily impacted by outliers, and I in fact ran into this very issue when I started running models. I ended up discovering that keeping all of the dummy features was blowing up the coefficients, and thereby the predictions and RSME.

# In[42]:


# What we're doing here is adding the dummy features for 
# one categorical feature at a time and running the regression.
dummy_cols = [col for col in model_data.columns if col not in num_features]
features_to_try = []
for cat in cat_features:
    cat_dummies = [c for c in dummy_cols if c.startswith(cat)]
    features_to_try += cat_dummies
    X_train_subset = X_train[num_features + boolean_features + features_to_try]
    X_test_subset = X_test[num_features + boolean_features + features_to_try]
    
    lr = LinearRegression()
    lr.fit(X_train_subset, y_train)
    
    print('Dummy Features: {} | Train RSME: {:.3f} | Test RSME: {:.3f}'.format(
        len(features_to_try), rsme(lr, X_train_subset, y_train).min(), rsme(lr, X_test_subset, y_test).min()))


# ### Ridge Regression
# Ridge Regression is an L2 penalized model where the squared sum of the weights are added to the OLS cost function.

# In[43]:


# We're using GridSearch here to find the optimal alpha value
param_grid = {'alpha': [0.01, 0.1, 1., 5., 10., 25., 50., 100.]}
ridge = GridSearchCV(Ridge(), cv=5, param_grid=param_grid, scoring='neg_mean_squared_error')
ridge.fit(X_train, y_train)
alpha = ridge.best_params_['alpha']

# Hone in
param_grid = {'alpha': [x/100. * alpha for x in range(50, 150, 5)]}
ridge = GridSearchCV(Ridge(), cv=5, param_grid=param_grid, scoring='neg_mean_squared_error')
ridge.fit(X_train, y_train)
alpha = ridge.best_params_['alpha']
ridge = ridge.best_estimator_

print('Ridge -> Train RSME: {:.5f} | Test RSME: {:.5f} | alpha: {:.5f}'.format(
    rsme(ridge, X_train, y_train).mean(), rsme(ridge, X_test, y_test).mean(), alpha))


# ### Plotly Dash Dashboard for Model Results

# In[44]:


# Create a function to generate the plot and return the image as a base64 string
def generate_plot(model, X_train, y_train, X_test, y_test):
    y_train_preds = model.predict(X_train)
    y_test_preds = model.predict(X_test)

    plt.figure(figsize = (12, 6))
    
    # Residuals
    plt.subplot(121)
    plt.scatter(y_train_preds, y_train_preds - y_train, c = 'blue', marker = 'o', label = 'Training data')
    plt.scatter(y_test_preds, y_test_preds - y_test, c = 'orange', marker = 's', label = 'Validation data')
    plt.title('Residuals')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc = 'upper left')
    plt.hlines(y = 0, xmin = y_train.min(), xmax = y_train.max(), color = 'red')

    # Predictions
    plt.subplot(122)
    plt.scatter(y_train_preds, y_train, c = 'blue', marker = 'o', label = 'Training data')
    plt.scatter(y_test_preds, y_test, c = 'orange', marker = 's', label = 'Validation data')
    plt.title('Predictions')
    plt.xlabel('Predicted values')
    plt.ylabel('Real values')
    plt.legend(loc = 'upper left')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], c = 'red')
    plt.tight_layout()
    
    # Save the figure to a BytesIO object
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format = 'png')
    img_buffer.seek(0)
    
    # Convert the image to a base64 string
    base64_image = base64.b64encode(img_buffer.read()).decode()
    
    return base64_image

# Generate model evaluation plots
model_evaluation_image = generate_plot(ridge, X_train, y_train, X_test, y_test)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Ames Iowa Dashboard"),
    
    # Display the model evaluation plot using an HTML img tag
    html.Img(src = 'data:image/png;base64,{}'.format(model_evaluation_image), width = '80%'),
    
    # Add more components as needed
])

if __name__ == '__main__':
    app.run_server(mode = 'inline', debug = True)

