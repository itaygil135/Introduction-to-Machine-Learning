import ast
import sys
from itertools import combinations

from sklearn.svm import SVR


import numpy as np
import pandas
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, \
    precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

fish_set = set()
not_fish_set = set()


fish_set = set()
not_fish_set = set()

fish_neg = set()
fish_pos = set()
not_fish_neg = set()
not_fish_pos = set()

posDict = dict()
negDict = dict()

combined_dict = {}
fishNotFishDict = {}

neg_combinations = ['neg', 'nEg', 'neG', 'NEg', 'Neg', 'nEG', 'NeG', 'NEG']
pos_combinations = ['pos', 'pOs', 'poS', 'POs', 'Pos', 'pOS', 'PoS', 'POS']


def tag_fish_dict():
    for pos_fish in fish_pos:
        if "+" in pos_fish:
            idx = pos_fish.index("+")
            pos_fish = " " + pos_fish + " "
            if pos_fish[idx + 1 - 1].isnumeric():
                posDict[pos_fish] = int(pos_fish[idx + 1 - 1])
            elif pos_fish[idx + 1 + 1].isnumeric():
                posDict[pos_fish] = int(pos_fish[idx + 1 + 1])
            else:
                posDict[pos_fish] = 2
        else:
            posDict[pos_fish] = 2
    # print("posDict:")
    # print(posDict)
    for neg_fish in fish_neg:
        if "+" in neg_fish:
            idx = neg_fish.index("+")
            neg_fish = " " + neg_fish + " "
            if neg_fish[idx + 1 - 1].isnumeric():
                negDict[neg_fish] = int(neg_fish[idx + 1 - 1]) * -1
            elif neg_fish[idx + 1 + 1].isnumeric():
                negDict[neg_fish] = int(neg_fish[idx + 1 + 1]) * -1
            else:
                negDict[neg_fish] = 2 * -1
        else:
            negDict[neg_fish] = 2 * -1
    # print("negDict:")
    # print(negDict)


def fish_tag(words, value, posNeg):
    for word in words:
        if word in value:
            if posNeg == "neg":
                fish_neg.add(value)
            else:
                fish_pos.add(value)
            return
    fish_pos.add(value)


################################################################################################

nonFishPosDict = dict()
nonFishNegDict = dict()


def tag_not_fish_dict():
    for pos_not_fish in not_fish_pos:
        if "+" in pos_not_fish:
            idx = pos_not_fish.index("+")
            pos_not_fish = " " + pos_not_fish + " "
            if pos_not_fish[idx + 1 - 1].isnumeric():
                nonFishPosDict[pos_not_fish] = int(pos_not_fish[idx + 1 - 1])
            elif pos_not_fish[idx + 1 + 1].isnumeric():
                nonFishPosDict[pos_not_fish] = int(pos_not_fish[idx + 1 + 1])
            else:
                nonFishPosDict[pos_not_fish] = 0
        else:
            nonFishPosDict[pos_not_fish] = 0
    # print("nonFishPosDict:")
    # print(nonFishPosDict)
    for neg_not_fish in not_fish_neg:
        if "+" in neg_not_fish:
            idx = neg_not_fish.index("+")
            neg_not_fish = " " + neg_not_fish + " "
            if neg_not_fish[idx + 1 - 1].isnumeric():
                nonFishNegDict[neg_not_fish] = int(
                    neg_not_fish[idx + 1 - 1]) * -1
            elif neg_not_fish[idx + 1 + 1].isnumeric():
                nonFishNegDict[neg_not_fish] = int(
                    neg_not_fish[idx + 1 + 1]) * -1
            else:
                nonFishNegDict[neg_not_fish] = 0
        else:
            nonFishNegDict[neg_not_fish] = 0
    # print("nonFishNegDict:")
    # print(nonFishNegDict)


def not_fish_tag(words, value, posNeg):
    for word in words:
        if word in value:
            if posNeg == "neg":
                not_fish_neg.add(value)
            else:
                not_fish_pos.add(value)
            return
    not_fish_pos.add(value)


def split_fish():
    for fish in fish_set:
        fish_tag(neg_combinations, fish, "neg")
        if fish not in fish_neg:
            fish_tag(pos_combinations, fish, "pos")
    for not_fish in not_fish_set:
        not_fish_tag(neg_combinations, not_fish, "neg")
        if not_fish not in not_fish_neg:
            not_fish_tag(pos_combinations, not_fish, "pos")
    # print("-----neg not_fish:----")
    # print(not_fish_neg)
    # print("-----pos not_fish:----")
    # print(not_fish_pos)
    # print("-----neg fish:----")
    # print(fish_neg)
    # print("-----pos fish:----")
    # print(fish_pos)


def search_fish(words, value):
    for word in words:
        if word in value:
            fish_set.add(value)
            return
    not_fish_set.add(value)


def her2(col):
    words = ['fish', 'Fish', 'fIsh', 'FIsh', 'fiSh',
             'FiSh', 'fISh', 'FISh', 'fish', 'Fish', 'fIsh', 'FIsh', 'fiSh',
             'FiSh', 'FISh', 'FISH']
    for value in col:
        value = str(value)
        search_fish(words, value)
    # print(fish_set)
    # print("-----Ofek")
    # print(not_fish_set)

def duplicateAndMap(X):
    X['isFish'] = X['אבחנה-Her2']
    X['Her2Values'] = X['אבחנה-Her2']
    X.drop(['אבחנה-Her2'], inplace=True, axis=1)
    X['isFish'] = X['isFish'].map(fishNotFishDict)
    X['Her2Values'] = X['Her2Values'].map(combined_dict)
    X['isFish'].fillna(0, inplace=True)
    X['Her2Values'].fillna(0, inplace=True)

def parse(X):
    her2(X['אבחנה-Her2'].unique())
    split_fish()
    tag_fish_dict()
    tag_not_fish_dict()
    # print("--------------FishNotFishDict:----------------")
    fishNotFishDict.update({item: 1 for item in fish_set})
    fishNotFishDict.update({item: 0 for item in not_fish_set})
    # print(len(fishNotFishDict))
    # print("--------------ValuesDict:----------------")
    combined_dict.update(posDict)
    combined_dict.update(negDict)
    combined_dict.update(nonFishPosDict)
    combined_dict.update(nonFishNegDict)
    # print(combined_dict)
    # print(len(posDict))
    # print(len(negDict))
    # print(len(nonFishPosDict))
    # print(len(nonFishNegDict))
    # print(len(combined_dict))
    duplicateAndMap(X)
    return X




def get_uniq_labels():
    labels = pd.read_csv('train.labels.0.csv').to_numpy()
    uniq = set()
    for lst in labels:
        for sec_lst in lst:
            if sec_lst == '[]':
                continue
            sec_lst = sec_lst.split(',')

            for elem in sec_lst:
                x = list(elem)
                if '[' in x:
                    x.remove('[')
                if ']' in x:
                    x.remove(']')
                if ' ' in x:
                    x.remove(' ')
                x = ''.join(x)
                uniq.add(x)

    cleaned_data = [value.replace(" ", "") for value in uniq]
    unique_values = set(cleaned_data)
    num_unique_values = len(unique_values)
    return unique_values


def get_combinations(set_of_elements):
    combinations_list = []
    for size in range(1, 4):
        for combination in combinations(set_of_elements, size):
            combinations_list.append(combination)

    return combinations_list


def preproccess_labels(y):
    new_y = []

    def string_to_list(string):
        return ast.literal_eval(string)

    # Apply the function to each column
    y['אבחנה-Location of distal metastases'] = y[
        'אבחנה-Location of distal metastases'].apply(string_to_list)

    for i in y.to_numpy():
        new_y.append(i[0])
    return new_y


def find_diff_days(row):
    if pd.isna(row['אבחנה-Surgery date1']) or pd.isna(
            row['אבחנה-Diagnosis date']):
        return 99999
    else:
        return (row['אבחנה-Diagnosis date'].date() - row[
            'אבחנה-Surgery date1'].date()).days


def clean_unnecessary(X: pd.DataFrame, is_test=False, y0=None, y1=None):
    X.dropna()
    if is_test:
        X['labels0'] = y0
        X['labels1'] = y1
        X.drop_duplicates(subset='id-hushed_internalpatientid', inplace=True)
        y0 = X['labels0']
        y1 = X['labels1']
        X.drop(['labels0', 'labels1'], inplace=True, axis=1)
    X.drop([' Form Name', ' Hospital', 'User Name',
            'surgery before or after-Activity date',
            'surgery before or after-Actual activity',
            'id-hushed_internalpatientid', 'אבחנה-Surgery name1',
            'אבחנה-Surgery name2',
            'אבחנה-Surgery name3'], inplace=True, axis=1)

    X = X.loc[(X['אבחנה-Age'] > 0) &
              ((X['אבחנה-Basic stage'] == 'c - Clinical') | (
                      X['אבחנה-Basic stage'] == 'p - Pathological') | (
                       X['אבחנה-Basic stage'] == 'r - Reccurent') | (
                       X['אבחנה-Basic stage'] == 'Null'))]

    X['אבחנה-Diagnosis date'] = pd.to_datetime(X['אבחנה-Diagnosis date'],
                                               format="%d/%m/%Y %H:%M")

    X = X.loc[(1900 <= X['אבחנה-Diagnosis date'].dt.year) & (
            X['אבחנה-Diagnosis date'].dt.year <= 2023)]

    X['אבחנה-Basic stage'].replace({'p - Pathological': 2,
                                    'c - Clinical': 1, 'Null': 0,
                                    'r - Reccurent': 3},
                                   inplace=True)
    X['אבחנה-Histopatological degree'].replace(
        {'G2 - Modereately well differentiated': 2,
         'G1 - Well Differentiated': 1, 'Null': 0,
         'G3 - Poorly differentiated': 3},
        inplace=True)

    X['אבחנה-Histopatological degree'] = X[
        'אבחנה-Histopatological degree'].replace({'Null': 0,
                                                  'GX - Grade cannot be assessed': 0,
                                                  'G1 - Well Differentiated': 1,
                                                  'G2 - Modereately well differentiated': 2,
                                                  'G3 - Poorly differentiated': 3,
                                                  'G4 - Undifferentiated': 4})

    X['אבחנה-Ivi -Lymphovascular invasion'].replace({'0': 0, '+': 1,
                                                     'extensive': 1,
                                                     'yes': 1,
                                                     '(+)': 1,
                                                     'no': 0,
                                                     '(-)': 0,
                                                     'none': 0,
                                                     'No': 0,
                                                     'not': 0,
                                                     '-': 0,
                                                     'NO': 0,
                                                     'neg': 0,
                                                     'MICROPAPILLARY VARIANT': 1},
                                                    inplace=True)

    X['אבחנה-Ivi -Lymphovascular invasion'].where(
        ((X['אבחנה-Ivi -Lymphovascular invasion'] == 0) |
         (X['אבחנה-Ivi -Lymphovascular invasion'] == 1)), 0,
        inplace=True)
    X['אבחנה-KI67 protein'] = X['אבחנה-KI67 protein'].str.extract(r'(\d+)')
    X['אבחנה-KI67 protein'].fillna(0, inplace=True)
    X['אבחנה-KI67 protein'] = X['אבחנה-KI67 protein'].astype('int')
    X['אבחנה-KI67 protein'].where(X['אבחנה-KI67 protein'] <= 100, 100,
                                  inplace=True)
    X['אבחנה-Lymphatic penetration'].replace({'Null': 0,
                                              'L0 - No Evidence of invasion': 0,
                                              'LI - Evidence of invasion': 1,
                                              'L1 - Evidence of invasion of superficial Lym.': 2,
                                              'L2 - Evidence of invasion of depp Lym.': 3
                                              }, inplace=True)
    X['אבחנה-Ivi -Lymphovascular invasion'].where(
        ((X['אבחנה-Ivi -Lymphovascular invasion'] == 0) |
         (X['אבחנה-Ivi -Lymphovascular invasion'] == 1) |
         (X['אבחנה-Ivi -Lymphovascular invasion'] == 2) |
         (X['אבחנה-Ivi -Lymphovascular invasion'] == 3)), 0,
        inplace=True)
    X['אבחנה-M -metastases mark (TNM)'] = X[
        'אבחנה-M -metastases mark (TNM)'].str.extract(r'(\d+)')
    X['אבחנה-M -metastases mark (TNM)'].replace({np.nan: 0}, inplace=True)
    X['אבחנה-M -metastases mark (TNM)'] = X[
        'אבחנה-M -metastases mark (TNM)'].astype(int)

    X["אבחנה-Margin Type"].replace({"נקיים": 0, "ללא": 0, "נגועים": 1},
                                   inplace=True)

    X["אבחנה-N -lymph nodes mark (TNM)"] = X[
        "אבחנה-N -lymph nodes mark (TNM)"].str.extract(r'(\d+)')
    X["אבחנה-N -lymph nodes mark (TNM)"].replace({np.nan: 0}, inplace=True)
    X["אבחנה-N -lymph nodes mark (TNM)"] = X[
        "אבחנה-N -lymph nodes mark (TNM)"].astype(int)

    X["אבחנה-Side"].replace({"שמאל": 1, "ימין": 1, "דו צדדי": 2, np.nan: 0},
                            inplace=True)
    X["אבחנה-Side"] = X["אבחנה-Side"].astype(int)

    X["אבחנה-Stage"] = X["אבחנה-Stage"].str.extract(r'(\d+)')
    X["אבחנה-Stage"].replace({np.nan: 0}, inplace=True)
    X["אבחנה-Stage"] = X["אבחנה-Stage"].astype(int)

    X['אבחנה-Surgery sum'] = X['אבחנה-Surgery sum'].astype(float)
    X['אבחנה-Surgery sum'].replace({np.nan: 0}, inplace=True)

    X['אבחנה-T -Tumor mark (TNM)'] = X[
        'אבחנה-T -Tumor mark (TNM)'].str.extract(r'(\d)')
    X['אבחנה-T -Tumor mark (TNM)'].replace({np.nan: 0}, inplace=True)
    X['אבחנה-T -Tumor mark (TNM)'] = X['אבחנה-T -Tumor mark (TNM)'].astype(
        'int')
    # X = pd.get_dummies(X, prefix='histo_diagnosis_',
    #                    columns=['אבחנה-Histological diagnosis'])
    X.drop(['אבחנה-Histological diagnosis'], inplace = True, axis = 1)
    X['אבחנה-Diagnosis date'] = pd.to_datetime(X['אבחנה-Diagnosis date'],
                                               format="%d/%m/%Y %H:%M")
    X['אבחנה-Surgery date1'] = pd.to_datetime(X['אבחנה-Surgery date1'],
                                              format="%d/%m/%Y",
                                              errors='coerce')
    X['אבחנה-Nodes exam'].fillna(-1, inplace=True)
    X['אבחנה-Tumor width'].fillna(-1, inplace=True)
    X['אבחנה-Tumor depth'].fillna(-1, inplace=True)
    X['אבחנה-Positive nodes'].fillna(-1, inplace=True)

    # Calculate difference in days
    X['DateDiff'] = X.apply(find_diff_days, axis=1)
    X.drop(
        ['אבחנה-Diagnosis date', 'אבחנה-Surgery date1', 'אבחנה-Surgery date2',
         'אבחנה-Surgery date3'], inplace=True, axis=1)
    X.drop(['אבחנה-er', 'אבחנה-pr'], inplace=True, axis=1)
    X = parse(X)
    if is_test:
        return X, y0, y1
    return X


# def predict_metastases(train_X, train_y, test_X, test_y):
#     train_features = train_X
#     train_labels = train_y
#     # Preprocess the labels using MultiLabelBinarizer
#     label_binarizer = MultiLabelBinarizer()
#     train_labels_encoded = label_binarizer.fit_transform(
#         train_labels)
#
#     # Train the model
#     model = OneVsRestClassifier(LogisticRegression())
#     model.fit(train_features, train_labels_encoded)
#
#     # Make predictions for test features
#     test_features = X_test
#     test_predictions_encoded = model.predict(test_features)
#
#     # Convert the encoded predictions back to original labels
#     test_predictions = label_binarizer.inverse_transform(
#         test_predictions_encoded)
#
#     # Save the predictions to a CSV file
#     predictions_df = pd.DataFrame({'metastases_sites': test_predictions})
#     predictions_df.to_csv('test.predictions.csv', index=False)
#
#     # Print classification report for the training set
#     train_predictions_encoded = model.predict(train_features)
#     train_predictions = label_binarizer.inverse_transform(
#         train_predictions_encoded)
#     train_predictions_decoded = label_binarizer.inverse_transform(
#         train_labels_encoded)
#     classification_rep = classification_report(train_predictions_decoded,
#                                                train_predictions)
#     print(classification_rep)


def task_0(X_train, X_test, y_train, invers_dict, y_test=5):
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

    # Step 3: Train the model
    random_forest.fit(X_train, y_train)

    # Step 4: Evaluate the model
    y_pred = random_forest.predict(X_test)

    # Calculate evaluation metrics
    if type(y_test) != int:
        accuracy = accuracy_score(y_test, y_pred)
        f1_micro = f1_score(y_test, y_pred, average='micro')
        f1_macro = f1_score(y_test, y_pred, average='macro')

        print("Accuracy:", accuracy)
        print("F1_micro score:", f1_micro)
        print("F1_macro score:", f1_macro)
    else:
        y_pred = ["[]" if i == 231 else invers_dict[i] for i in y_pred]
        df = pd.DataFrame(y_pred).rename(columns={0:"אבחנה-Location of distal metastases"})
        df.to_csv('predictions0.csv', index=False)


def task_1(X_train, X_test, y_train, y_test=5):
    gradient_boosting = GradientBoostingRegressor()

    # Step 3: Train the model
    gradient_boosting.fit(X_train, y_train)

    # Step 4: Evaluate the model
    y_pred = gradient_boosting.predict(X_test)
    y_pred = [0 if i <0 else i for i in y_pred ]

    # Calculate evaluation metrics
    if type(y_test) != int:
        mse = mean_squared_error(y_test, y_pred)
        print("MSE: ", mse)
        r2 = r2_score(y_test, y_pred)
        return mse
    df = pd.DataFrame(y_pred).rename(columns={0:"אבחנה-Tumor size"})
    df.to_csv('predictions1.csv', index=False)
    # df = pd.DataFrame(y_pred)
    # df.to_csv('predictions2.csv', index=False)


def search_fish(words, value):
    for word in words:
        if word in value:
            fish_set.add(value)
            return
    not_fish_set.add(value)


def make_labels_dictionary(premotation):
    i = 0
    to_return = {}
    for prem in premotation:
        to_return[prem] = i
        i += 1
    to_return['[]'] = i
    return to_return


def labels_dictionary_inverse(dicti):
    inverse_dict = {}
    for key, val in dicti.items():
        inverse_dict[val] = list(sorted([t[1:-1] for t in key]))
    return inverse_dict


def change_label(y, label_dict):
    new_y = np.zeros(len(y))
    idx = 0
    for i in y:
        if len(i) == 0:
            new_y[idx] = label_dict['[]']
        else:
            new_i = [value.replace(" ", "") for value in i]

            for key, value in label_dict.items():
                if sorted(new_i) == sorted([t[1:-1] for t in key]):
                    new_y[idx] = value
        idx += 1
    return new_y