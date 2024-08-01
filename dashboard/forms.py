from django import forms 
from .models import Calories


class CaloriesForm(forms.ModelForm):
    class Meta:
        model = Calories
        fields = ['food_name', 'calories_amount', 'serving', 'category']

from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField()