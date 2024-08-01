from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('home/', views.index, name='dashboard-index'),
    path('user/', views.user, name='dashboard-user'),
    path('calories/', views.calories, name='dashboard-calories'),
    path('userinfo/', views.userinfo, name='dashboard-userinfo'),
    path('userinfo/<int:pk>/', views.userinfo_detail, name='dashboard-userinfo-detail'),
    path('predict/', views.predict, name='dashboard-predict'),
    path('calories/delete/<int:pk>/', views.delete_calories, name='dashboard-calories-delete'),
    path('calories/update/<int:pk>/', views.update_calorie, name='dashboard-calories-update'),
    path('caloriecounter/', views.caloriecounter, name='dashboard-caloriecounter'),
    path('dietrecomen/', views.recomend_diet, name='dashboard-dietrecomen'),
    path('caloriesburned/', views.predict_calories_burnt, name='dashboard-caloriesburnedpredict'),
    path('predictbodyfat/', views.predict_body_fat, name='dashboard-predict_body_fat'),
    path('predictimagefood/', views.predict_image, name='dashboard-predict_image'),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)