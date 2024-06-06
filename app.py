import pandas as pd,numpy as np,matplotlib.pyplot as plt,seaborn as sns
import streamlit as st
import scipy.stats as stats
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_percentage_error

@st.cache_data
def load_data():
    df = pd.read_csv('Concrete_Data_Yeh.csv')
    return df

df = load_data()

st.title('Concrete Regression')

st.write("check out this [link](https://www.kaggle.com/datasets/maajdl/yeh-concret-data)")

st.subheader('Head data')
st.dataframe(df.head())

st.subheader('Tail data')
st.dataframe(df.tail())

st.subheader('Missing values')
st.write(df.isna().sum())

st.subheader('Feature types')
st.write(df.dtypes)

st.subheader('Finding and removing outliers')
st.text('Using boxplot to find outliers')
fig = plt.figure(figsize=(14,10))
sns.boxplot(data=df)
st.pyplot(fig)

Q1 = df.quantile(q=0.25)
st.text('First quartile')
st.table(Q1)
Q3 = df.quantile(q=0.75)
st.text('Third quartile')
st.table(Q3)
IQR = df.apply(stats.iqr)
st.text("IQR")
st.table(IQR)

st.text("Shape of dataframe before removing outliers")
st.write(df.shape)

df_clean = df[~((df < (Q1 - 1.5*IQR)) | (df > (Q3 + 1.5*IQR))).any(axis=1)]

st.text("Shape of dataframe after removing outliers")
st.write(df_clean.shape)

st.subheader('Checking distribution')

st.text("Let's check skewness")

three_plot_fig = plt.figure(figsize=(15,30))
plt.subplot(211)
sns.distplot(df_clean['age'],kde=True)

plt.subplot(212)
sns.distplot(df_clean['slag'],kde=True)


st.pyplot(three_plot_fig)


st.text("The wholesome skewness of data is described in the table below")

df_clean_x = df_clean.iloc[:,:-1]
st.table(df_clean_x.skew())

pt = PowerTransformer(method='yeo-johnson')
data = pt.fit_transform(df_clean_x)

data_x = pd.DataFrame(data)

st.text("After removing skewness")

st.table(data_x.skew())

y = df_clean.iloc[:,-1]

st.subheader('Splitting training and testing data and creating model')
code = '''

x_train,x_test,y_train,y_test = tts(data,y,test_size=0.30,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)
'''
st.code(code, language='python')
x_train,x_test,y_train,y_test = tts(data,y,test_size=0.30,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)

st.subheader('Predicting the model')

code = '''
y_pred = model.predict(x_test)

'''
st.code(code, language='python')
y_pred = model.predict(x_test)

st.subheader('Computing score and error')
score = r2_score(y_test,y_pred)
mape= mean_absolute_percentage_error(y_test,y_pred)

st.text(f'r2 score is {score}')
st.text(f'error percentage is {mape}')
