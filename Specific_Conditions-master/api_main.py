import pickle

import pandas as pd
from flask import Flask, request
from numpy import array
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


@app.route('/')
def hello_world():
    result = 'Hello!\nWhat you passed isn\'t correct... O wrong nau...\nA lot of time was spent making this API so check your url and send the correct variables.\n\n  Yours truly, Bonus Salus\n✌'
    diabetes = str(request.args.get('diabetes'))
    cvd = str(request.args.get('cvd'))
    hypertension = str(request.args.get('hypertension'))
    stress = str(request.args.get('stress'))

    if diabetes == '1':
        pregnancies = str(request.args.get('pregnancies'))
        glucose = str(request.args.get('glucose'))
        bloodpressure = str(request.args.get('bloodpressure'))
        skinthickness = str(request.args.get('skinthickness'))
        insulin = str(request.args.get('insulin'))
        bmi = str(request.args.get('bmi'))
        diabetespedigreefunction = str(request.args.get('diabetespedigreefunction'))
        age = str(request.args.get('age'))

        dataset = pd.read_csv('/home/TheHealthDome2/Conditions/Data/Diabetes.csv')
        col = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
               'Age']
        X = dataset[list(col)].values

        # test = [[6, 148, 72, 35, 155.54822, 33.6, 0.627, 50]]
        test = [
            [float(pregnancies), float(glucose), float(bloodpressure), float(skinthickness), float(insulin), float(bmi),
             float(diabetespedigreefunction), float(age)]]
        sc = StandardScaler()
        X = sc.fit_transform(X)
        test = sc.transform(test)

        filename = '/home/TheHealthDome2/Conditions/Models/Diabetes.pickle'
        loaded_model = pickle.load(open(filename, 'rb'))

        result = loaded_model.predict(test)

    if cvd == '1':

        age = str(request.args.get('age'))
        gender = str(request.args.get('gender'))
        height = str(request.args.get('height'))
        weight = str(request.args.get('weight'))
        ap_hi = str(request.args.get('ap_hi'))
        ap_lo = str(request.args.get('ap_lo'))
        chol = str(request.args.get('chol'))
        glucs = str(request.args.get('gluc'))
        smoke = str(request.args.get('smoke'))
        alco = str(request.args.get('alco'))
        active = str(request.args.get('active'))

        df = pd.read_csv('/home/TheHealthDome2/Conditions/Data/CVD.csv', sep=';')
        df['age'] = (df['age'] / 365).round().astype('int')
        df.drop(['id'], axis=1, inplace=True)
        df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
        df.isnull().values.any()
        df[df.ap_lo > df.ap_hi]
        df.sort_values('BMI')
        df.drop(df[(df['height'] > df['height'].quantile(0.99)) | (df['height'] < df['height'].quantile(0.01))].index,
                inplace=True)
        df.drop(df[(df['weight'] > df['weight'].quantile(0.99)) | (df['weight'] < df['weight'].quantile(0.01))].index,
                inplace=True)
        df.drop(df[(df['ap_hi'] > df['ap_hi'].quantile(0.99)) | (df['ap_hi'] < df['ap_hi'].quantile(0.01))].index,
                inplace=True)
        df.drop(df[(df['ap_lo'] > df['ap_lo'].quantile(0.99)) | (df['ap_lo'] < df['ap_lo'].quantile(0.01))].index,
                inplace=True)
        df = df[df.ap_hi >= df.ap_lo]
        df.drop(df[(df['BMI'] > df['BMI'].quantile(0.99)) | (df['BMI'] < df['BMI'].quantile(0.01))].index, inplace=True)
        values = array(df['cholesterol'])
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        a = []
        b = []
        c = []
        for i in onehot_encoded:
            a.append(i[0])
            b.append(i[1])
            c.append(i[2])
        df['chol1'] = a
        df['chol2'] = b
        df['chol3'] = c
        df.drop(['cholesterol'], axis=1, inplace=True)
        gluc = pd.get_dummies(df['gluc'], prefix='gluc')
        df = pd.concat([df, gluc], axis=1)
        df.drop(['gluc'], axis=1, inplace=True)
        df['BMI'] = df['BMI'].round().astype('int')
        df['chol1'] = df['chol1'].astype('int')
        df['chol2'] = df['chol2'].astype('int')
        df['chol3'] = df['chol3'].astype('int')
        cols = df.columns.drop(['cardio'])
        df.drop(df[df.duplicated(cols)].index, inplace=True)
        df['gender'].replace(2, 0, inplace=True)
        down = df[(df.cardio == 1) & (df.active == 1) & (df.smoke == 0) & (df.gluc_1 == 1) & (df.BMI < 25) & (
                    80 < df.ap_lo) & (df.ap_lo < 90) & (120 < df.ap_hi) & (df.ap_hi < 140) & (df.chol1 == 1)].index
        df.drop(down, inplace=True)
        down2 = df[(df.cardio == 0) & (df.active == 0) & (df.gluc_3 == 1) & (df.BMI > 30) & (
                    (90 < df.ap_lo) | (df.ap_lo < 80)) & ((140 < df.ap_hi) | (df.ap_hi < 120)) & (df.chol3 == 1)].index
        df.drop(down2, inplace=True)
        df[(df.cardio == 0) & (df.active == 0) & (df.BMI > 35) & ((90 < df.ap_lo) | (df.ap_lo < 80)) & (
                    (140 < df.ap_hi) | (df.ap_hi < 120)) & ((df.chol3 == 1) | (df.gluc_3 == 1))]
        df.drop(df[(df.BMI > 40) & (df.cardio == 0)].index, inplace=True)
        df.groupby('gender').count()
        down3 = df[(df.gluc_3 == 1) & (df.chol3 == 1) & (df.cardio == 0) & ((df.ap_hi > 140) | (df.ap_hi < 120))].index
        df.drop(down3, inplace=True)
        dflow0 = df[
            (df.age < 45) & (df.active == 1) & (df.chol1 == 1) & (df.cardio == 0) & (df.smoke == 0) & (df.alco == 0) & (
                        df.gluc_1 == 1) & (df.BMI <= 35)]
        dflow1 = df[(df.age < 45) & (df.active == 0) & (df.chol1 == 0) & (df.gluc_1 == 0)]
        dfmid0 = df[(df.cardio == 0) & (df.age >= 45) & (df.age <= 58) & ((df.chol1 == 1) & (df.active == 1)) & (
                    df.gluc_1 == 1) & (df.smoke == 0) & (df.alco == 0) & (df.BMI <= 30)]
        dfmid1 = df[(df.cardio == 1) & (df.age >= 45) & (df.age <= 58) & ((df.chol1 == 0) | (df.active == 0))]
        dfhi1 = df[(df.age > 60) & (df.cardio == 1) & (df.chol1 == 0)]
        dfhi0 = df[
            (df.age > 60) & (df.cardio == 0) & (df.chol1 == 1) & (df.smoke == 0) & (df.alco == 0) & (df.gluc_1 == 1) & (
                        df.BMI <= 30)]
        final = dflow0.append([dflow1, dfmid0, dfmid1, dfhi0, dfhi1])
        bad = df[(df.cardio == 0) & (df.active == 0) & (df.chol1 == 0)]
        bad1 = df[(df.cardio == 1) & (df.active == 1) & (df.chol1 == 1)]
        newfinal = final.append(bad)
        newfinal1 = newfinal.append(bad1)
        final = newfinal1
        predictors = final.drop(["cardio", 'BMI'], axis=1)
        target = final['cardio']
        X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.2, random_state=1,
                                                            stratify=target)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1,
                                                          stratify=Y_train)

        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_val = sc_X.fit_transform(X_val)

        def get_input(age, gender, height, weight, ap_hi, ap_lo, chol, gluc, smoke, alco, active):
            if chol == 1:
                chol1 = 1
                chol2 = 0
                chol3 = 0
            elif chol == 2:
                chol1 = 0
                chol2 = 1
                chol3 = 0
            else:
                chol1 = 0
                chol2 = 0
                chol3 = 1

            if gluc == 1:
                gluc1 = 1
                gluc2 = 0
                gluc3 = 0
            elif gluc == 2:
                gluc1 = 0
                gluc2 = 1
                gluc3 = 0
            else:
                gluc1 = 0
                gluc2 = 0
                gluc3 = 1

            return [age, gender, height, weight, ap_hi, ap_lo, smoke, alco, active, chol1, chol2, chol3, gluc1, gluc2,
                    gluc3]

        # test = get_input(56, 1, 166, 80, 160, 100, 3, 3, 1, 1, 3)
        test = get_input(age, gender, height, weight, ap_hi, ap_lo, chol, glucs, smoke, alco, active)

        X_test = sc_X.fit_transform([test])

        filename = '/home/TheHealthDome2/Conditions/Models/CVD.pickle'
        new_gb = pickle.load(open(filename, 'rb'))
        result = new_gb.predict(X_test)

    if hypertension == '1':

        sex = str(request.args.get('sex'))
        familyhxht = str(request.args.get('familyhxht'))
        poor = str(request.args.get('poor '))
        bmi = str(request.args.get('bmi '))
        wc = str(request.args.get('wc'))
        sbp = str(request.args.get('sbp'))
        dbp = str(request.args.get('dbp'))
        htn = str(request.args.get('htn'))

        filename = '/home/TheHealthDome2/Conditions/Models/Hypertension.pickle'
        tree = pickle.load(open(filename, 'rb'))
        # example1 = [2, 2, 0, 24.143655, 92, 110, 80, 0, ]
        example1 = [sex, familyhxht, poor, bmi, wc, sbp, dbp, htn]
        example = pd.Series(data=example1, index=['Sex', 'FamilyHxHT', 'Poor ', 'BMI ', 'WC', 'SBP', 'DBP', 'HTN'])

        def classify_example(example, tree):
            question = list(tree.keys())[0]
            feature_name, comparison_operator, value = question.split(" ")

            # ask question
            if example[feature_name] <= value:
                answer = tree[question][0]
            else:
                answer = tree[question][1]

            # base case
            if not isinstance(answer, dict):
                return answer

            # recursive part
            else:
                residual_tree = answer
                return classify_example(example, residual_tree)

        results = int(classify_example(example, tree))
        result = [results]

    if stress == '1':
        ecg = str(request.args.get('ecg'))
        emg = str(request.args.get('emg'))
        foot_gsr = str(request.args.get('foot_gsr'))
        hand_gsr = str(request.args.get('hand_gsr'))
        hr = str(request.args.get('hr'))
        resp = str(request.args.get('resp'))

        df = pd.read_csv('/home/TheHealthDome2/Conditions/Data/Stress.csv', header=None)

        df.columns = ['Target', 'ECG(mV)', 'EMG(mV)', 'Foot GSR(mV)', 'Hand GSR(mV)', 'HR(bpm)', 'RESP(mV)']
        X_train, X_test, y_train, y_test = train_test_split(
            df[['ECG(mV)', 'EMG(mV)', 'Foot GSR(mV)', 'Hand GSR(mV)', 'HR(bpm)', 'RESP(mV)']], df['Target'],
            test_size=0.30, random_state=12345)
        filename = "/home/TheHealthDome2/Conditions/Models/Stress.pickle"

        minmax_scale = preprocessing.MinMaxScaler().fit(
            df[['ECG(mV)', 'EMG(mV)', 'Foot GSR(mV)', 'Hand GSR(mV)', 'HR(bpm)', 'RESP(mV)']])
        df_minmax = minmax_scale.transform(
            df[['ECG(mV)', 'EMG(mV)', 'Foot GSR(mV)', 'Hand GSR(mV)', 'HR(bpm)', 'RESP(mV)']])
        X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(df_minmax, df['Target'], test_size=0.30,
                                                                                random_state=42)

        knn_norm = KNeighborsClassifier(n_neighbors=5)
        filename = '/home/TheHealthDome2/Conditions/Models/Stress.pickle'
        knn_norm = pickle.load(open(filename, 'rb'))

        # pred_data_norm = minmax_scale.transform([[0.001,0.931,5.91,19.773,99.065,35.59]])
        pred_data_norm = minmax_scale.transform([[ecg, emg, foot_gsr, hand_gsr, hr, resp]])
        pred = knn_norm.predict(pred_data_norm)
        result = str(pred)

    return str(result)
