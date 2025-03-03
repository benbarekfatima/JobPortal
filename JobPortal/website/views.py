from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.urls import reverse
from django.contrib.auth.models import User, Group
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from datetime import datetime
from django.contrib.admin.views.decorators import staff_member_required
from django.core.mail import send_mail
from django.contrib.auth.models import Group
from .models import Employer, Employee, Job, Application
from .forms import UserForm, EmployerSignUpForm, EmployeeSignUpForm, UserLoginForm, EmployeeProfileForm, EmployerProfileForm
import datetime
import torch
from .recommend.graph import add_node_user,add_node_job,modify_node_user,modify_node_job,add_edge_app,delete_edge_app
from .recommend.preprocess import preprocess_resume,preprocess_text
from .recommend.recommeder import recommend_top_k
from .recommend import global_vars


def home(request):
    context = {'message': 'Welcome to your Website App!'}
    return render(request, 'home.html', context)

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
            #login(request, user)
            # Add the user to an 'Employer' group
            employer_group, _ = Group.objects.get_or_create(name='Employer')
            employer_group.user_set.add(user)
            return HttpResponseRedirect(reverse('waitforvalidation')) 
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
            # add user to graph 
            resume_file = employee.resume.path
            text,topic = preprocess_resume(resume_file)
            data=torch.load(global_vars.graph_path)
            data=add_node_user(data, userID=employee.employee_id, text=text, topic=topic)
            torch.save(data,global_vars.graph_path)
            print(data)
            print(data['user'])
            # Add the user to an 'Employee' group
            employee_group, _ = Group.objects.get_or_create(name='Employee')
            employee_group.user_set.add(user)
            return HttpResponseRedirect(reverse('display_jobs', args=[employee.employee_id]))  # Use employee.employee_id instead of employee.id
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
        required_applicants = request.POST.get('required_applicants')  # Add this line to get the number of required applicants

        employer = Employer.objects.get(employer_id = employer_id)
        new_job = Job(title = title, employer=employer, description=desc, location=location, requirements=requirements, salary=salary,
                    closing_date=closing_date, required_applicants=required_applicants)
        new_job.save()
        # add job to graph
        text_job= title + " "+desc+" "+requirements
        text,topic = preprocess_text(text_job)
        data=torch.load(global_vars.graph_path)
        data=add_node_job(data, jobID=new_job.job_id, text=text, topic=topic)
        torch.save(data,global_vars.graph_path)
        print(data['job'])

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
        job.required_applicants = request.POST.get('required_applicants')  # Add this line to handle required_applicants
        job.save()
        # modify in graph
        text_job= job.title + " "+job.description+" "+job.requirements
        text,topic = preprocess_text(text_job)
        data=torch.load(global_vars.graph_path)
        data=modify_node_job(data,job_id,text,topic)
        torch.save(data,global_vars.graph_path)
        print(data)
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
    jobs.reverse()
    data = []
    for job in jobs:
        employer = Employer.objects.get(employer_id = job['employer_id'])
        data.append((job, employer.name))
    return render(request, 'employeeJobBoard.html', {"jobs":data, "applicant": applicant})


def applyToJob(request, applicant, job_id):
    # Fetch the employee
    employee = get_object_or_404(Employee, employee_id=applicant)
    # Fetch all jobs with their employer names
    jobs = list(Job.objects.all().values())
    job_data = [(job, Employer.objects.get(employer_id=job['employer_id']).name) for job in jobs]
    # Initialize success and error flags
    success = False
    error = False
    try:
        # Try to get an existing application
        application = Application.objects.get(job_id=job_id, applicant=employee)
        print('You already applied for this job')
        error = True
    except Application.DoesNotExist:
        # Create a new application if none exists
        application = Application(job_id=job_id, applicant=employee, application_date=datetime.datetime.now(), status='submitted')
        application.save()
        success = True
        # Add application to graph
        graph_data = torch.load(global_vars.graph_path)
        graph_data = add_edge_app(graph_data, applicant, job_id)
        torch.save(graph_data, global_vars.graph_path)
        print(graph_data)
    # Render the job board with updated application status
    return render(request, 'employeeJobBoard.html', {"jobs": job_data, "applicant": applicant, "success": success, "error": error})


def displayJobRecommandations(request, applicant):
    # Load the model and data
    data = torch.load(global_vars.graph_path)
    print(data['user'])
    print(data['job'])
    model = global_vars.model
    num_jobs = Job.objects.count()
    top_k_job_ids = recommend_top_k(applicant, data, model, k=num_jobs)
    available_job_ids = set(Job.objects.filter(job_id__in=top_k_job_ids).values_list('job_id', flat=True))
    ordered_job_ids = [job_id for job_id in top_k_job_ids if job_id in available_job_ids]
    top_10_job_ids = ordered_job_ids[:10]
    ordered_jobs = [Job.objects.get(job_id=job_id) for job_id in top_10_job_ids]

    data = []
    for job in ordered_jobs:
        employer = Employer.objects.get(employer_id=job.employer.employer_id)
        data.append((job, employer.name))
    print(data)
    employee = Employee.objects.get(employee_id=applicant)  # Fetch the employee object
    return render(request, 'recommandations.html', {"jobs": data, "applicant": applicant, "employee": employee})


def viewJobDetails(request, applicant, job_id):
    try:
        job = Job.objects.get(job_id = job_id)
    except Job.DoesNotExist:
        return render(request,'error.html')
    employee = Employee.objects.get(employee_id=applicant)  # Fetch the employee object
    return render(request, 'viewJobDetails.html', {"job":job, "applicant":applicant, "employee":employee})


def display_employee_profile(request, applicant):
    employee = get_object_or_404(Employee, pk=applicant)
    return render(request, 'employee_profile.html', {'employee': employee})

def edit_employee_profile(request, applicant):
    employee = get_object_or_404(Employee, pk=applicant)
    if request.method == 'POST':
        form = EmployeeProfileForm(request.POST, request.FILES, instance=employee)
        if form.is_valid():
            form.save()
            # modify in graph
            resume_file = employee.resume.path
            text,topic = preprocess_resume(resume_file)
            data=torch.load(global_vars.graph_path)
            data=modify_node_user(data,employee.employee_id,text,topic)
            torch.save(data,global_vars.graph_path)
            print(data)

            return redirect('employee_profile', applicant=applicant)
    else:
        form = EmployeeProfileForm(instance=employee)
    return render(request, 'edit_employee_profile.html', {'form': form, 'employee': employee})

def display_employer_profile(request, employer_id):
    employer = get_object_or_404(Employer, pk=employer_id)
    return render(request, 'employer_profile.html', {'employer': employer})

def edit_employer_profile(request, employer_id):
    employer = get_object_or_404(Employer, pk=employer_id)
    if request.method == 'POST':
        form = EmployerProfileForm(request.POST, instance=employer)
        if form.is_valid():
            form.save()
            return redirect('employer_profile', employer_id=employer_id)
    else:
        form = EmployerProfileForm(instance=employer)
    return render(request, 'edit_employer_profile.html', {'form': form, 'employer': employer})

def acceptJobApplication(request, employer_id, job_id, application_id, applicant):
    try:
        application = Application.objects.get(id=application_id, applicant=applicant)
        application.status = 'accepted'
        application.save()

        # Check if the job has reached the required number of accepted applicants
        job = Job.objects.get(job_id=job_id)
        accepted_applications_count = Application.objects.filter(job=job, status='accepted').count()
        if accepted_applications_count >= job.required_applicants:
            # If the required number of accepted applicants is reached, delete the job
            job.delete()

    except (Application.DoesNotExist, Job.DoesNotExist):
        print("Error: Application or Job does not exist.")

    # Redirect back to the list of jobs for the employer
    return HttpResponseRedirect(reverse('jobs', args=[employer_id]))


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

def waitforvalidation(request):
    return render(request, 'waitforvalidation.html')
