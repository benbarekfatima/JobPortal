 
from django.contrib import admin
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
  path('home',views.home,name='home'),
  path('employerSignUp/', views.employer_signup, name='employer-signup'),
  path('employeeSignUp/', views.employee_signup, name='employee-signup'),
  path('login/', views.user_login, name='login'),
  path('profile/<int:user>', views.display_profile, name='profile'),
  path('logout/', views.user_logout, name='logout'),
  path('jobs/<int:employer_id>', views.displayJobsEmployer, name='jobs'),
  path('jobs/<int:employer_id>/new', views.addNewJob, name='new_job'),
  path('jobs/<int:employer_id>/edit/<int:job_id>', views.editJob, name='edit_job'),
  path('jobs/<int:employer_id>/status/<int:job_id>/<int:application_id>/<int:applicant>/accept', views.acceptJobApplication, name='accept_job_application'),
  path('jobs/<int:employer_id>/status/<int:job_id>/<int:application_id>/<int:applicant>/refuse', views.refuseJobApplication, name='refuse_job_application'),
  path('jobs/<int:employer_id>/status/<int:job_id>/<int:application_id>/<int:applicant>/delete', views.deleteJobApplication, name='delete_job_application'),
  path('jobs/<int:employer_id>/delete/<int:job_id>', views.deleteJob, name='delete_job'),
  path('jobs/<int:employer_id>/<int:job_id>/candidates', views.displayApplications, name='display_applications'),
  path('jobs/applicant/<int:applicant>', views.displayJobs, name='display_jobs'),
  path('jobs/applicant/<int:applicant>/<int:job_id>/view', views.viewJobDetails, name='view_job_details'),
  path('jobs/applicant/<int:applicant>/<int:job_id>', views.applyToJob, name='apply_job'),
  path('jobs/applicant/<int:applicant>/recommandations', views.displayJobRecommandations, name='job_recommandation'),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
