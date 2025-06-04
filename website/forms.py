from django import forms
from django.forms import DateInput
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
import re
from . import models
from django.forms import ModelChoiceField
from .models import Employee, Employer, Job, Application
from django import forms 


class UserForm(UserCreationForm):
    email = forms.EmailField(validators=[validate_email], required=True)
    username = forms.CharField(max_length=150, required=True)
    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=True)

    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'username', 'email']

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise ValidationError('This email address is already in use.')
        return email

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if User.objects.filter(username=username).exists():
            raise ValidationError('This username is already in use.')
        return username
    
class EmployerSignUpForm(forms.ModelForm):
    phone_number = forms.CharField(max_length=20)
    location = forms.CharField(max_length=255)
    industry = forms.CharField(max_length=255)
    bio = forms.CharField(max_length=255)

    class Meta:
        model = models.Employer
        fields = ['phone_number', 'location', 'industry', 'bio']


class EmployeeSignUpForm(forms.ModelForm):
    phone_number = forms.CharField(max_length=20)
    location = forms.CharField(max_length=255)
    skills = forms.CharField(widget=forms.Textarea, required=False)
    link = forms.URLField(required=False)
    major = forms.CharField(max_length=255)
    degree = forms.CharField(max_length=255)
    resume = forms.FileField()

    class Meta:
        model =  models.Employee
        fields = ['phone_number', 'location', 'skills', 'link', 'major','degree', 'resume']


class UserLoginForm(forms.Form):
    username = forms.CharField(label='Enter your username', max_length=150)
    password = forms.CharField(label='Enter your password', widget=forms.PasswordInput())
    

class EmployerProfileForm(forms.ModelForm):
    class Meta:
        model = Employer
        fields = ['name', 'email', 'phone_number', 'location', 'industry', 'bio']

class EmployeeProfileForm(forms.ModelForm):
    class Meta:
        model = Employee
        fields = ['name', 'email','phone_number', 'location', 'skills', 'link', 'major', 'degree', 'resume']

