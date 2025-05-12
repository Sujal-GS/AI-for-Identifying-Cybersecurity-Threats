import pandas as pd
import numpy as np
import pickle as pkls
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

Friday_WorkingHours_Afternoon_DDos = pd.read_csv(
    r"D:\Capstone project\CICIDS2017Dataset\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
Friday_WorkingHours_Afternoon_PortScan = pd.read_csv(
    r"D:\Capstone project\CICIDS2017Dataset\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
Friday_WorkingHours_Morning = pd.read_csv(
    r"D:\Capstone project\CICIDS2017Dataset\Friday-WorkingHours-Morning.pcap_ISCX.csv")
Monday_WorkingHours = pd.read_csv(
    r"D:\Capstone project\CICIDS2017Dataset\Monday-WorkingHours.pcap_ISCX.csv")
Thursday_WorkingHours_Afternoon_Infilteration = pd.read_csv(
    r"D:\Capstone project\CICIDS2017Dataset\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
Thursday_WorkingHours_Morning_WebAttacks = pd.read_csv(
    r"D:\Capstone project\CICIDS2017Dataset\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
Tuesday_WorkingHours = pd.read_csv(
    r"D:\Capstone project\CICIDS2017Dataset\Tuesday-WorkingHours.pcap_ISCX.csv")
Wednesday_workingHours = pd.read_csv(
    r"D:\Capstone project\CICIDS2017Dataset\Wednesday-workingHours.pcap_ISCX.csv")  
df = pd.concat([Friday_WorkingHours_Afternoon_DDos, Friday_WorkingHours_Afternoon_PortScan, Friday_WorkingHours_Morning, Monday_WorkingHours,
               Thursday_WorkingHours_Afternoon_Infilteration, Thursday_WorkingHours_Morning_WebAttacks, Tuesday_WorkingHours, Wednesday_workingHours], axis=0)
df.columns = Friday_WorkingHours_Afternoon_DDos.columns
df[' Label'] = df[' Label'].apply(
    lambda x: 'BENIGN' if x == 'BENIGN' else 'ATTACK')
encoder = LabelEncoder()
df[' Label'] = encoder.fit_transform(df[' Label'])
df = df.fillna(0)  # Replace NaN with 0
df = df.replace([np.inf, -np.inf], 0)
df = df.astype(int)
x = df.drop(' Label', axis=1)
y = df[' Label']
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

imputer = SimpleImputer(strategy='mean')
x_imputed = imputer.fit_transform(x)

num_columns = df.shape[1]

k = min(20, num_columns)  
k_best = SelectKBest(score_func=f_classif, k=k)
x_new = k_best.fit_transform(x_imputed, y)
selected_features_mask = k_best.get_support()
print(selected_features_mask)
elected_feature_names = x.columns[selected_features_mask]
new_columns = [' Flow Duration', 'Bwd Packet Length Max', ' Bwd Packet Length Min', ' Bwd Packet Length Mean', ' Bwd Packet Length Std', ' Flow IAT Std', ' Flow IAT Max', 'Fwd IAT Total', ' Fwd IAT Std', ' Fwd IAT Max',
               ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance', ' Average Packet Size', ' Avg Bwd Segment Size', 'Idle Mean', ' Idle Max', ' Idle Min']
df_new = x[new_columns]
df_new['label'] = df[' Label']
x1 = df_new.iloc[:, :-1].values
y1 = df_new.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(
    x1, y1, test_size=0.3, random_state=42)
ann = Sequential()
ann.add(Dense(units=20, activation='relu', input_shape=(x_train.shape[1],)))
ann.add(Dense(units=20, activation='relu'))  
ann.add(Dense(units=1, activation='sigmoid'))
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

ann.fit(x_train, y_train, batch_size=32, epochs=50, callbacks=[early_stopping])
test_input = np.array([3268, 72, 72, 0, 0, 0, 0, 201, 72,
                      32, 3268, 72, 72, 0, 0, 0, 0, 201, 72, 32]).reshape(1, -1)
print(ann.predict(test_input))
with open('project_pkl', 'rb') as f:
     ann = pkls.load(f)
out = ann.predict(x_test)
out = out.round()
test = pd.DataFrame(out)
test[0] = test[0].apply(lambda x: 'BENIGN' if x == 0 else 'ATTACK')
print(test[0].unique())

test.to_csv("output.csv", index=False)
print("Output extracted to CSV")
print("Project Done")