"""legal_advisor URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.index),
    path('contact/',views.contact),
    path('userReg/',views.Userregister),
    path('login/',views.logins),
    path('AdvocateReg/',views.AdvocateReg),
    path('Regpay/',views.Regpay),
    path('userhome/',views.userhome),
    path('advocatehome/',views.advocatehome),
    path('admhome/',views.admhome),
    path('admviewusers/',views.admviewusers),
    path('ipc_sections/',views.ipc_sections),
    path('admviewadvocate/',views.admviewadvocate),
    path('add_case_category/',views.add_case_category),
    path('addipc_section/',views.addipc_section),
    path('userview_Advocate/',views.userview_Advocate),
    path('user_home/',views.user_home),
    path('user_add_feed/',views.user_add_feed),
    path('user_view_ipc/',views.user_view_ipc),
    path('user_book_case/',views.user_book_case),
    path('user_view_request/',views.user_view_request),
    path('adv_view_request/',views.adv_view_request),
    path('approve_case_request/',views.approve_case_request),
    path('reject_case_request/',views.reject_case_request),
    path('approve_advocate/',views.approve_advocate),
    path('reject_advocate/',views.reject_advocate),
    path('advocate_add_feed/',views.advocate_add_feed),
    path('adv_view_ipc/',views.adv_view_ipc),
    path('adv_view_approved_case/',views.adv_view_approved_case),
    path('user_profile/',views.user_profile),
    path('user_case_files/',views.user_case_files),
    path('chat/',views.chat),
    path('payment_page/',views.payment_page),
    path('payment_view/',views.payment_view),
    path('Applied_success/',views.Applied_success),
    path('userchat/',views.userchat),
    path('reply/',views.reply),
    path('admviewfeedback/',views.admviewfeedback),
    path('asdfcasdcas/',views.udp),
    path('delete_feedback/',views.delete_feedback),
    path('add_rating/',views.add_rating),
]
