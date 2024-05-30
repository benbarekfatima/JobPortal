from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.models import User, Group
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from datetime import datetime
from django.contrib.admin.views.decorators import staff_member_required
from django.core.mail import send_mail
from .models import Employer, Employee, Job, Application
from .forms import UserForm, EmployerSignUpForm, EmployeeSignUpForm, UserLoginForm
import datetime


def home(request):
    context = {'message': 'Welcome to your Website App!'}
    return render(request, 'home.html', context)



from django.contrib.auth.models import Group

def employer_signup(request):
    if request.method == 'POST':
        user_form = UserForm(request.POST)
        employer_form = EmployerSignUpForm(request.POST)
        if user_form.is_valid() and employer_form.is_valid():
            user = user_form.save()
            employer = employer_form.save(commit=False)
            employer.user = user
            employer.name = user.username
            employer.email = user.email
            employer.save()
            login(request, user)
            # Add the user to an 'EMPLOYER' group
            employer_group, _ = Group.objects.get_or_create(name='Employer')
            employer_group.user_set.add(user)
            return render(request, 'redirectionPage.html')
    else:
        user_form = UserForm()
        employer_form = EmployerSignUpForm()

    return render(request, 'employersignup.html', {'user_form': user_form, 'employer_form': employer_form})


def employee_signup(request):
    if request.method == 'POST':
        user_form = UserForm(request.POST)
        employee_form = EmployeeSignUpForm(request.POST, request.FILES)
        if user_form.is_valid() and employee_form.is_valid():
            user = user_form.save()
            employee = employee_form.save(commit=False)
            employee.user = user
            employee.name = user.username
            employee.email = user.email
            employee.save()
            login(request, user)
            employee_group= Group.objects.get(name='Employee')
            employee_group.user_set.add(user)
            return render(request, 'redirectionPage.html')
    else:
        user_form = UserForm()
        employee_form = EmployeeSignUpForm()

    return render(request, 'employeesignup.html', {'user_form': user_form, 'employee_form': employee_form})


def user_login(request):
    if request.method == 'POST':
        login_form = UserLoginForm(request.POST)
        if login_form.is_valid():
            username = login_form.cleaned_data['username']
            password = login_form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                user_id = user.id
                login(request, user)
                if user.groups.filter(name='Employee').exists():
                    employee_id = Employee.objects.get(user_id = user_id)
                    employee_id = employee_id.employee_id
                    return HttpResponseRedirect(f'/project/jobs/applicant/{employee_id}')
                elif user.groups.filter(name='Employer').exists():
                    employer_id = Employer.objects.get(user_id = user_id)
                    employer_id = employer_id.employer_id
                    return HttpResponseRedirect(f'/project/jobs/{employer_id}')
                elif user.is_staff:
                    return HttpResponseRedirect('/project/admin/')
                else:
                    return render(request,'waitforvalidation.html')
    else:
        login_form = UserLoginForm()

    return render(request, 'login.html', {'login_form': login_form})

def user_logout(request):
    logout(request)
    return HttpResponseRedirect('/project/login/')


def getListOfJobs():
    jobs = list(Job.objects.all().values())
    data = []
    for job in jobs:
        employer = Employer.objects.get(employer_id = job['employer_id'])
        data.append((job, employer.name))
    return data

def displayJobsEmployer(request, employer_id):
    list_of_jobs = list(Job.objects.filter(employer=employer_id).values())
    print('list', list_of_jobs)
    context={
        'jobs':list_of_jobs,
        'employer':employer_id
    }
    return render(request, 'employerJobBoard.html', {"data": context})

def addNewJob(request, employer_id):
    if(request.method == 'POST'):
        title = request.POST.get('job_title')
        desc = request.POST.get('job_description')
        salary = request.POST.get('salary')
        requirements = request.POST.get('requirements')
        location = request.POST.get('location')
        closing_date = request.POST.get('closing-date')
        employer = Employer.objects.get(employer_id = employer_id)
        new_job = Job(title = title, employer=employer, description=desc, location=location, requirements=requirements, salary=salary,
                    closing_date=closing_date)
        new_job.save()
        return HttpResponseRedirect(f'/project/jobs/{employer_id}')
    else:
        return render(request, 'addnewjob.html', {'employer_id':employer_id})
    
def editJob(request, employer_id, job_id):
    if(request.method=='POST'):
        job = Job.objects.get(job_id=job_id)
        job.title = request.POST.get('job_title')
        job.description = request.POST.get('job_description')
        job.salary = request.POST.get('salary')
        job.requirements = request.POST.get('requirements')
        job.location = request.POST.get('location')
        job.closing_date = request.POST.get('closing-date')
        job.save()
        return HttpResponseRedirect(f'/project/jobs/{employer_id}')
    else:
        job = Job.objects.get(job_id=job_id)
        return render(request, "editJob.html", {'employer_id':employer_id, 'job_id':job_id, 'job':job})
    
def deleteJob(request, job_id, employer_id):
    job = Job.objects.get(job_id=job_id)
    job.delete()
    return HttpResponseRedirect(f'/project/jobs/{employer_id}')

def displayApplications(request, employer_id, job_id):
    applications = Application.objects.filter(job = job_id)
    job_title = Job.objects.get(job_id=job_id)
    job_title = job_title.title
    return render(request, 'displayEmployerApplications.html', {'data':applications,'employer_id':employer_id, 'job_id':job_id, 'position':job_title})

def displayJobs(request, applicant):
    jobs = list(Job.objects.all().values())
    data = []
    for job in jobs:
        employer = Employer.objects.get(employer_id = job['employer_id'])
        data.append((job, employer.name))
    return render(request, 'employeeJobBoard.html', {"jobs":data, "applicant": applicant})

def applyToJob(request, applicant, job_id):
    employee = Employee.objects.get(employee_id = applicant)
    jobs = list(Job.objects.all().values())
    data = []
    for job in jobs:
        employer = Employer.objects.get(employer_id = job['employer_id'])
        data.append((job, employer.name))
    try:
        application = Application.objects.get(job_id=job_id, applicant=employee)
        print('you already applied for this job')
        error=True
        success=False
        # return HttpResponseRedirect(f'/project/jobs/applicant/{applicant}')
        return render(request, 'employeeJobBoard.html', {"jobs":data, "applicant": applicant, "success":success, "error":error})
    except Application.DoesNotExist:
        application = Application(job_id=job_id, applicant=employee, application_date=datetime.datetime.now(), status='submitted')
        application.save()
        success = True
        error=False
        return render(request, 'employeeJobBoard.html', {"jobs":data, "applicant": applicant, "success":success, "error":error})
        
def displayJobRecommandations(request, applicant):
    jobs_list = list(Job.objects.filter(category = "Tech").values())
    print('jobs', jobs_list)
    data=[]
    for job in jobs_list:
        employer = Employer.objects.get(employer_id = job['employer_id'])
        data.append((job, employer.name))
    return render(request, 'recommandations.html', {"jobs":data})

def viewJobDetails(request, applicant, job_id):
    try:
        job = Job.objects.get(job_id = job_id)
    except Job.DoesNotExist:
        return render(request,'error.html')
    return render(request, 'viewJobDetails.html', {"job":job, "applicant":applicant})


def display_profile(request, user):
    return render(request, 'profile.html')

def acceptJobApplication(request,employer_id, job_id, application_id, applicant):
    try:
        application = Application.objects.get(id = application_id, applicant = applicant)
        application.status = 'accepted'
        application.save()
    except Application.DoesNotExist:
        print("error")
    return HttpResponseRedirect(f'/project/jobs/{employer_id}/{job_id}/candidates')


def refuseJobApplication(request,employer_id, job_id, application_id, applicant):
    try:
        application = Application.objects.get(id = application_id, applicant = applicant)
        application.status = 'refused'
        application.save()
    except Application.DoesNotExist:
        print("error")
    return HttpResponseRedirect(f'/project/jobs/{employer_id}/{job_id}/candidates')

def deleteJobApplication(request,employer_id, job_id, application_id, applicant):
    try:
        application = Application.objects.get(id = application_id, applicant = applicant)
        application.delete()
    except Application.DoesNotExist:
        print("error")
    return HttpResponseRedirect(f'/project/jobs/{employer_id}/{job_id}/candidates')
