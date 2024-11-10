from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    address= models.CharField(max_length=3000)
    gender= models.CharField(max_length=30)

class price_prediction(models.Model):

    RID= models.CharField(max_length=3000)
    Car_Name= models.CharField(max_length=3000)
    Location= models.CharField(max_length=3000)
    Car_Year= models.CharField(max_length=3000)
    kilometer= models.CharField(max_length=3000)
    Fuel_Type= models.CharField(max_length=3000)
    Transmission= models.CharField(max_length=3000)
    Owner_Type= models.CharField(max_length=3000)
    Mileage= models.CharField(max_length=3000)
    Engine= models.CharField(max_length=3000)
    Power= models.CharField(max_length=3000)
    Seats= models.CharField(max_length=3000)
    Prediction= models.CharField(max_length=3000)


class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



