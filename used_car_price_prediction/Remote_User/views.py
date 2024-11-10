from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import openpyxl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# Create your views here.
from Remote_User.models import ClientRegister_Model,price_prediction,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Register1(request):
    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city, address=address, gender=gender)
        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html', {'object': obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def predict_used_car_price_type(request):
    if request.method == "POST":

        RID= request.POST.get('RID')
        Car_Name= request.POST.get('Car_Name')
        Location= request.POST.get('Location')
        Car_Year= request.POST.get('Car_Year')
        kilometer= request.POST.get('kilometer')
        Fuel_Type= request.POST.get('Fuel_Type')
        Transmission= request.POST.get('Transmission')
        Owner_Type= request.POST.get('Owner_Type')
        Mileage= request.POST.get('Mileage')
        Engine= request.POST.get('Engine')
        Power= request.POST.get('Power')
        Seats= request.POST.get('Seats')

        df = pd.read_csv('Datasets.csv')
        df
        df.columns

        def apply_results(results):

            if float(results) <= 5.0:
                return 0  # Price is Below 5L
            elif float(results) >= 5.0 and float(results) <= 20.0:
                return 1  # More Than 5 and less than 20
            elif float(results) >= 20.0 and float(results) <= 100.0:
                return 2  # Price is More than 20L and Less that 100

        df['Results'] = df['Price'].apply(apply_results)

        cv = CountVectorizer()
        X = df['RID'].apply(str)
        y = df['Results']

        X = cv.fit_transform(X)


        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("KNeighborsClassifier")
        from sklearn.neighbors import KNeighborsClassifier
        kn = KNeighborsClassifier()
        kn.fit(X_train, y_train)
        knpredict = kn.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, knpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, knpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, knpredict))
        models.append(('KNeighborsClassifier', kn))

        # SVM Model
        print("SVM")
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression
        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))

        print("Random Forest Classifier")
        from sklearn.ensemble import RandomForestClassifier
        rf_clf = RandomForestClassifier()
        rf_clf.fit(X_train, y_train)
        rfpredict = rf_clf.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, rfpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, rfpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, rfpredict))
        models.append(('RandomForestClassifier', rf_clf))

        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        RID1 = [RID]
        vector1 = cv.transform(RID1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if prediction == 0:
            val = 'Below 5L'
        elif prediction == 1:
            val = 'More Than 5L and Below 20L'
        elif prediction == 2:
            val = 'More Than 20L and Below 100L'


        print(val)
        print(pred1)

        price_prediction.objects.create(
        RID=RID,
        Car_Name=Car_Name,
        Location=Location,
        Car_Year=Car_Year,
        kilometer=kilometer,
        Fuel_Type=Fuel_Type,
        Transmission=Transmission,
        Owner_Type=Owner_Type,
        Mileage=Mileage,
        Engine=Engine,
        Power=Power,
        Seats=Seats,
        Prediction=val,
        )

        return render(request, 'RUser/predict_used_car_price_type.html',{'objs': val})
    return render(request, 'RUser/predict_used_car_price_type.html')



