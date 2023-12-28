import pandas as pd
import numpy as np
import pickle


data=pd.read_csv("train.csv")

data.duplicated().sum()


data["previous_year_rating"]=data["previous_year_rating"].replace(np.NaN,data["previous_year_rating"].mean())
data['education'].fillna(data['education'].mode()[0], inplace = True)


from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
label_encoder = LabelEncoder()
oe = OrdinalEncoder()

education_column_reshaped = data['education'].values.reshape(-1, 1)
data['education'] = oe.fit_transform(education_column_reshaped)

data['region'] = label_encoder.fit_transform(data['region'])

data['department'] = label_encoder.fit_transform(data['department'])

data['recruitment_channel'] = label_encoder.fit_transform(data['recruitment_channel'])

data['gender'] = label_encoder.fit_transform(data['gender'])

data=data.drop('employee_id',axis=1)

data.info()

X = data[['avg_training_score', 'no_of_trainings', 'KPIs_met >80%', 'length_of_service', 'awards_won?','previous_year_rating']]
y = data['is_promoted']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(X_train)
x_test  = sc.transform(X_test)
x_train=pd.DataFrame(x_train)
x_test=pd.DataFrame(x_test)

from sklearn.neighbors import KNeighborsClassifier
metric_k=[]
neighbors=np.arange(3,15)
for k in neighbors:
  classifier=KNeighborsClassifier(n_neighbors=k,metric='euclidean',p=2)
  classifier.fit(x_train,y_train)
  y_prediction=classifier.predict_proba(x_test)


pickle.dump(classifier,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))

