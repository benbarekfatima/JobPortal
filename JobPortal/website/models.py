from django.db import models
from django.contrib.auth.models import User


class Employer(models.Model):
  employer_id = models.AutoField(primary_key=True)  # Auto-incrementing integer as ID
  user = models.OneToOneField(User, on_delete=models.CASCADE, unique=True, null=False)
  name = models.CharField(max_length=255)
  email = models.EmailField(unique=True) 
  phone_number = models.CharField(max_length=20, blank=True)  # Optional phone number
  location = models.CharField(max_length=255)
  industry = models.CharField(max_length=255)
  bio = models.CharField(max_length=255)

  def __str__(self):
    return self.name
  
def validate_file_extension(value):
  """
  Custom validator to ensure uploaded file is a PDF.
  """
  import os
  from django.core.exceptions import ValidationError
  allowed_extensions = ['pdf']
  extension = os.path.splitext(value.name)[1].lower()
  extension = extension.split('.')[1]
  if extension not in allowed_extensions:
    raise ValidationError(f"Unsupported file extension. Only PDFs are allowed.")
  
class Employee(models.Model):
  employee_id = models.AutoField(primary_key=True)  # Auto-incrementing integer as ID
  user = models.OneToOneField(User, on_delete=models.CASCADE, unique=True, null=False)
  name = models.CharField(max_length=255)
  email = models.EmailField(unique=True)  # Unique email for each employee
  phone_number = models.CharField(max_length=20, blank=True)  # Optional phone number
  skills = models.TextField(blank=True)  # Text field for listing skills
  location = models.CharField(max_length=255)
  link = models.URLField(blank=True)
  major = models.CharField(max_length=255)
  degree = models.CharField(max_length=255)
  resume = models.FileField(upload_to='employee_resumes/', validators=[validate_file_extension])  # Upload field for PDF resume

  def __str__(self):
    return self.name

class Job(models.Model):
  job_id = models.AutoField(primary_key=True)  # Auto-incrementing integer as ID
  title = models.CharField(max_length=255)
  employer = models.ForeignKey(Employer, on_delete=models.CASCADE)  # Foreign key to Employer
  description = models.TextField()
  location = models.CharField(max_length=255)
  requirements = models.TextField()  # Text field for listing job requirements
  salary = models.CharField(max_length=255, blank=True)  # Optional salary range or specific amount
  closing_date = models.DateField()  # Application deadline
  category = models.CharField(max_length=255, default="")
  required_applicants = models.PositiveIntegerField(default=1)  # Number of applicants needed for the job


  def __str__(self):
    return self.title
  
class Application(models.Model):
  # application_id = models.AutoField(primary_key=True)  # Auto-incrementing integer as ID
  job = models.ForeignKey(Job, on_delete=models.CASCADE)  # Foreign key to Job
  applicant = models.ForeignKey(Employee, on_delete=models.CASCADE)  # Foreign key to Employee
  # resume = models.FileField(upload_to='resumes/')  # Upload field for storing resumes
  # cover_letter = models.TextField(blank=True)  # Optional cover letter text
  application_date = models.DateTimeField(auto_now_add=True)  # Automatically set on application creation
  status = models.CharField(max_length=50, choices=[('submitted', 'Submitted'), ('accepted', 'Accepted'), ('rejected', 'Rejected')])

  def __str__(self):
    return f"{self.applicant.name} applied for {self.job.title}"