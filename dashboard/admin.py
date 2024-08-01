from django.contrib import admin

from .models import Calories

from django.contrib.auth.models import Group


admin.site.site_header = 'Rifki Administrator Dashboard'

class CaloriesAdmin(admin.ModelAdmin):
    list_display = ('food_name', 'calories_amount', 'serving', 'category')
    list_filter = ['category']


# Register your models here.



admin.site.register(Calories, CaloriesAdmin)
# admin.site.unregister(Group)

