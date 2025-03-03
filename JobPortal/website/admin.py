from django.contrib import admin
from .models import Employer, Employee, Job, Application
# Register your models here.
admin.site.register(Employer)
admin.site.register(Employee)
admin.site.register(Job)
admin.site.register(Application)

