

from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from .models import Calories
from datetime import datetime, timedelta
from django.contrib.auth.models import User
from .forms import CaloriesForm
from django.contrib import messages
from django.db.models import Sum
from collections import defaultdict
from django.db.models import Sum
from datetime import datetime, timedelta
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
import langchain.globals as lc_globals
from django.http import JsonResponse
import os
import re

from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
import langchain.globals as lc_globals

# Configure verbose mode
lc_globals.set_verbose(True)

# Set up Google API key and model
os.environ["GOOGLE_API_KEY"] = ''
generation_config = {"temperature": 0.6, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
model = GoogleGenerativeAI(model="gemini-pro", generation_config=generation_config)

prompt_template_resto = PromptTemplate(
    input_variables=['age', 'gender', 'weight', 'height', 'veg_or_nonveg', 'disease', 'region', 'state', 'allergics', 'foodtype'],
    template="Diet Recommendation System:\n"
             "I want you to recommend 5 restaurants names, 5 breakfast names with calories intake per 100 gram service, 5 dinner names with calories detail per 100 gram service, and 5 workout names with burned calories detail, "
             "based on the following criteria:\n"
             "Person age: {age}\n"
             "Person gender: {gender}\n"
             "Person weight: {weight}\n"
             "Person height: {height}\n"
             "Person veg_or_nonveg: {veg_or_nonveg}\n"
             "Person generic disease: {disease}\n"
             "Person region: {region}\n"
             "Person state: {state}\n"
             "Person allergics: {allergics}\n"
             "Person foodtype: {foodtype}."
)
chain_resto = LLMChain(llm=model, prompt=prompt_template_resto)

@login_required(login_url='user-login')
def recomend_diet(request):
    if request.method == 'POST':
        # Extract form data
        age = request.POST.get('age')
        gender = request.POST.get('gender')
        weight = request.POST.get('weight')
        height = request.POST.get('height')
        veg_or_nonveg = request.POST.get('veg_or_nonveg')
        disease = request.POST.get('disease')
        region = request.POST.get('region')
        state = request.POST.get('state')
        allergics = request.POST.get('allergics')
        foodtype = request.POST.get('foodtype')

        # Define the input dictionary
        input_data = {
            'age': age,
            'gender': gender,
            'weight': weight,
            'height': height,
            'veg_or_nonveg': veg_or_nonveg,
            'disease': disease,
            'region': region,
            'state': state,
            'allergics': allergics,
            'foodtype': foodtype
        }

        try:
            # Invoke the chain and get results
            response = chain_resto.invoke(input_data)
            
            # Ensure response contains text
            if isinstance(response, dict):
                results_text = response.get('text', '')
            else:
                results_text = str(response)

            # Debug output
            print("Results Text:", results_text)

            # Function to clean unwanted characters at the beginning of the text
            def clean_text(text):
                # Remove leading stars or asterisks and other unwanted characters
                cleaned_text = re.sub(r'^\*+', '', text).strip()
                return cleaned_text

            # Clean the response text
            cleaned_text = clean_text(results_text)

            # Split the cleaned_text into sections based on known patterns
            sections = re.split(r'(\n\n)', cleaned_text)
            context = {
    'input_data': input_data,
    'results': cleaned_text
            }

            # Prepare context for rendering
            return render(request, 'dashboard/dietrecomen.html', context)

        except Exception as e:
            print("Error:", str(e))
            return render(request, 'dashboard/dietrecomen.html', {'error': str(e)})

    return render(request, 'dashboard/dietrecomen.html')



@login_required(login_url='user-login')
def index(request):
    return render(request, 'dashboard/index.html')



@login_required(login_url='user-login')
def user(request):
    # Get today's date without time part
    today = datetime.today().date()

    # Filter and aggregate today's food items
    calories_today = Calories.objects.filter(
        user=request.user, date__date=today)
    food_dict = defaultdict(int)
    for calorie in calories_today:
        food_dict[calorie.food_name] += calorie.calories_amount

    # Prepare data for the pie chart
    food_names = list(food_dict.keys())
    calories_amount = list(food_dict.values())

    # Calculate total calorie intake per day for the last 7 days
    days = 7  # Define the number of days to show in the bar chart
    date_today = datetime.today().date()
    date_start = date_today - timedelta(days=days)

    calorie_per_day = (
        Calories.objects.filter(user=request.user, date__date__range=[
                                date_start, date_today])
        .values('date__date')
        .annotate(total_calories=Sum('calories_amount'))
        .order_by('date__date')
    )

    # Prepare data for the bar and line chart
    dates = [entry['date__date'].strftime(
        '%Y-%m-%d') for entry in calorie_per_day]
    total_calories = [entry['total_calories'] for entry in calorie_per_day]
    daily_changes = [total_calories[i] - total_calories[i - 1] if i >
                     0 else total_calories[i] for i in range(len(total_calories))]
    daily_changes = [max(0, change)
                     for change in daily_changes]  # Ensure no negative values

    context = {
        'food_names': food_names,
        'calories_amount': calories_amount,
        'dates': dates,
        'total_calories': total_calories,
        'daily_changes': daily_changes
    }

    return render(request, 'dashboard/user.html', context)


@login_required(login_url='user-login')
def userinfo(request):
    users = User.objects.all()
    total_users = users.count()  # Get the total number of users

    context = {
        'users': users,
        'total_users': total_users,  # Pass the total number to the template
    }
    return render(request, 'dashboard/userinfo.html', context)



@login_required(login_url='user-login')
def userinfo_detail(request, pk):
    user = get_object_or_404(User, pk=pk)

    context = {
        'user_detail': user,
    }
    return render(request, 'dashboard/userinfo_detail.html', context)




@login_required(login_url='user-login')
def calories(request):
    items = Calories.objects.filter(user=request.user)
    today = datetime.today().date()
    items_today = items.filter(date__date=today)
    items_previous = items.exclude(date__date=today)
    total_calories_today = sum(item.calories_amount for item in items_today)

    # Data for the pie chart
    food_dict = defaultdict(int)
    for calorie in items_today:
        food_dict[calorie.food_name] += calorie.calories_amount
    food_names = list(food_dict.keys())
    calories_amount = list(food_dict.values())

    # Data for the mixed chart
    days = 7
    date_today = datetime.today().date()
    date_start = date_today - timedelta(days=days)
    calorie_per_day = (
        Calories.objects.filter(user=request.user, date__date__range=[
                                date_start, date_today])
        .values('date__date')
        .annotate(total_calories=Sum('calories_amount'))
        .order_by('date__date')
    )
    dates = [entry['date__date'].strftime('%Y-%m-%d') for entry in calorie_per_day]
    total_calories = [entry['total_calories'] for entry in calorie_per_day]
    daily_changes = [total_calories[i] - total_calories[i - 1] if i > 0 else total_calories[i] for i in range(len(total_calories))]
    daily_changes = [max(0, change) for change in daily_changes]

    if request.method == 'POST':
        form = CaloriesForm(request.POST)
        if form.is_valid():
            calorie_entry = form.save(commit=False)
            calorie_entry.user = request.user
            calorie_entry.date = datetime.now()
            calorie_entry.save()
            Calories_name = form.cleaned_data.get('food_name')
            messages.success(request, f'{Calories_name} has been added')
            return redirect('dashboard-calories')
    else:
        form = CaloriesForm()

    context = {
        'items_today': items_today,
        'items_previous': items_previous,
        'total_calories_today': total_calories_today,
        'form': form,
        'food_names': food_names,
        'calories_amount': calories_amount,
        'dates': dates,
        'total_calories': total_calories,
        'daily_changes': daily_changes,
    }
    return render(request, 'dashboard/calories.html', context)



@login_required(login_url='user-login')
def delete_calories(request, pk):
    item = get_object_or_404(Calories, pk=pk, user=request.user)
    if request.method == 'POST':
        item.delete()
        return redirect('dashboard-calories')
    return render(request, 'dashboard/delete_calorie.html', {'item': item})


@login_required(login_url='user-login')
def update_calorie(request, pk):
    item = get_object_or_404(Calories, pk=pk, user=request.user)
    if request.method == 'POST':
        form = CaloriesForm(request.POST, instance=item)
        if form.is_valid():
            form.save()
            return redirect('dashboard-calories')
    else:
        form = CaloriesForm(instance=item)

    context = {
        'form': form,
        'item': item
    }
    return render(request, 'dashboard/update_calorie.html', context)


@login_required(login_url='user-login')
def predict(request):
    return render(request, 'dashboard/predict.html')

# def adminside(request):
#     return render(request, 'dashboard/adminside.html')
    

# Add other views as needed

from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import requests
import json

@login_required(login_url='user-login')
def caloriecounter(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        api_url = 'https://api.calorieninjas.com/v1/nutrition?query='
        headers = {'X-Api-Key': ''}
        
        try:
            api_request = requests.get(api_url + query, headers=headers)
            api_request.raise_for_status()  # Check for HTTP request errors
            api_response = api_request.json()  # Parse the JSON response

            if api_response.get('items'):  # Check if there are items in the response
                return render(request, 'dashboard/caloriecounter.html', {'api_response': api_response})
            else:
                return render(request, 'dashboard/caloriecounter.html', {'message': 'I\'m sorry, I can\'t find it for you. Try searching using another word or per component on your food (e.g egg)'})

        except requests.exceptions.RequestException as e:
            # Handle request errors
            return render(request, 'dashboard/caloriecounter.html', {'error': 'API request failed: ' + str(e)})
        except json.JSONDecodeError:
            # Handle JSON parsing errors
            return render(request, 'dashboard/caloriecounter.html', {'error': 'Error parsing the API response.'})
    
    # If the request method is GET, simply render the form
    return render(request, 'dashboard/caloriecounter.html')



# MACHINE LEARNING 
# predict 1
import pickle
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required

# Load the model when the Django server starts
pipeline = None
try:
    with open('MachineLearning/Predictions_Calories_Burned/Model/burntpredictmodel.pkl', 'rb') as f:
        pipeline = pickle.load(f)
except Exception as e:
    print(f"Error loading the model: {e}")

@login_required(login_url='user-login')
def predict_calories_burnt(request):
    prediction = None
    if request.method == 'POST':
        if pipeline is None:
            return JsonResponse({'error': 'Model not loaded properly'}, status=500)
        
        try:
            gender = request.POST.get('gender')
            age = float(request.POST.get('age'))
            height = float(request.POST.get('height'))
            weight = float(request.POST.get('weight'))
            duration = float(request.POST.get('duration'))
            heart_rate = float(request.POST.get('heart_rate'))
            body_temp = float(request.POST.get('body_temp'))

            sample = pd.DataFrame({
                'Gender': [gender],
                'Age': [age],
                'Height': [height],
                'Weight': [weight],
                'Duration': [duration],
                'Heart_Rate': [heart_rate],
                'Body_Temp': [body_temp],
            }, index=[0])

            result = pipeline.predict(sample)
            prediction = float(result[0])
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return render(request, 'dashboard/caloriesburnedpredict.html', {'prediction': prediction})


# MACHINE LEARNING 2 

from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import pickle

def predict_body_fat(request):
    if request.method == 'POST':
        try:
            # Load the transformer and model
            transformer = pickle.load(open('MachineLearning/BodyFat_Predictions/Model/transformer.pkl', 'rb'))
            model = pickle.load(open('MachineLearning/BodyFat_Predictions/Model/model.pkl', 'rb'))

            # Get data from POST request
            data = {
                'Age': float(request.POST['age']),
                'Weight': float(request.POST['weight']),
                'Height': float(request.POST['height']),
                'Neck': float(request.POST['neck']),
                'Chest': float(request.POST['chest']),
                'Abdomen': float(request.POST['abdomen']),
                'Hip': float(request.POST['hip']),
                'Thigh': float(request.POST['thigh']),
                'Knee': float(request.POST['knee']),
                'Ankle': float(request.POST['ankle']),
                'Biceps': float(request.POST['biceps']),
                'Forearm': float(request.POST['forearm']),
                'Wrist': float(request.POST['wrist']),
            }

            # Feature engineering
            bmi = 703 * data['Weight'] / (data['Height'] ** 2)
            acr = data['Abdomen'] / data['Chest']
            htr = data['Hip'] / data['Thigh']

            x = pd.DataFrame([{
                'Age': data['Age'],
                'Neck': data['Neck'],
                'Knee': data['Knee'],
                'Ankle': data['Ankle'],
                'Biceps': data['Biceps'],
                'Forearm': data['Forearm'],
                'Wrist': data['Wrist'],
                'Bmi': bmi,
                'ACratio': acr,
                'HTratio': htr
            }])

            # Transform features
            x_transformed = transformer.transform(x)

            # Predict density
            density = model.predict(x_transformed)[0]

            # Calculate body fat percentage
            body_fat_percentage = ((4.95 / density) - 4.5) * 100

            return render(request, 'dashboard/predict_body_fat.html', {'predicted_body_fat': body_fat_percentage})

        except Exception as e:
            return render(request, 'dashboard/predict_body_fat.html', {'error': str(e)})

    else:
        return render(request, 'dashboard/predict_body_fat.html')
    


# Image predictions and scraping 
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from .forms import ImageUploadForm
import numpy as np
import requests
from bs4 import BeautifulSoup

import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.core.files.storage import default_storage
from PIL import Image
from django.conf import settings

# Ensure the media directory exists
import os
import numpy as np
import tensorflow as tf
import requests
from bs4 import BeautifulSoup
from django.shortcuts import render
from django.core.files.storage import default_storage
from PIL import Image
from django.conf import settings

os.makedirs(settings.MEDIA_ROOT, exist_ok=True)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'MachineLearning', 'FoodImages_Predictions', 'Model', 'food_model3.tflite')
labels_path = os.path.join(BASE_DIR, 'MachineLearning', 'FoodImages_Predictions', 'Model', 'food_label.txt')


interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Muat label
with open(labels_path, 'r') as f:
    labels = {int(line.split(': ')[1]): line.split(': ')[0] for line in f.readlines()}

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0  # Ubah tipe data menjadi FLOAT32
    img = np.expand_dims(img, axis=0)
    return img

def scrape_nutrition_data(predicted_dish):
    base_url = 'https://www.fatsecret.co.id/kalori-gizi/umum/'
    query = predicted_dish.replace(' ', '-')
    url = base_url + query

    # Kirim permintaan HTTP GET ke URL
    response = requests.get(url)

    # Pastikan permintaan berhasil
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Temukan div yang berisi fakta nutrisi
        nutrition_facts = soup.find('div', {'class': 'nutrition_facts international'})

        if nutrition_facts:
            data = {}

            # Ukuran Sajian
            serving_size_label = nutrition_facts.find('div', {'class': 'serving_size_label'}).text
            serving_size_value = nutrition_facts.find('div', {'class': 'serving_size_value'}).text
            data[serving_size_label] = serving_size_value

            # Informasi Nutrisi
            nutrient_divs = nutrition_facts.find_all('div', {'class': 'nutrient'})
            i = 0
            while i < len(nutrient_divs) - 1:
                nutrient_name = nutrient_divs[i].text.strip()
                nutrient_value = nutrient_divs[i+1].text.strip()
                if nutrient_name and nutrient_value:
                    data[nutrient_name] = nutrient_value
                i += 2

            # Ambil tabel rincian kalori
            breakdown_table = soup.find('table', {'class': 'generic spaced'})
            breakdown_details = None
            if breakdown_table:
                rows = breakdown_table.find_all('tr')
                for row in rows:
                    if 'Kalori Rincian' in row.text:
                        breakdown_details = row.find('div', {'class': 'small'}).text.strip()
                        break

            return data, breakdown_details
        else:
            return None, None
    else:
        return None, None

def predict_image(request):
    if request.method == 'POST' and 'image' in request.FILES:
        image = request.FILES['image']
        image_path = default_storage.save(os.path.join('tmp', image.name), image)
        image_full_path = os.path.join(settings.MEDIA_ROOT, image_path)
        image_url = os.path.join(settings.MEDIA_URL, image_path)

        img = preprocess_image(image_full_path)

        # Set tensor untuk menunjuk ke data input yang akan diinferensikan
        interpreter.set_tensor(input_details[0]['index'], img)
        
        # Jalankan inferensi
        interpreter.invoke()
        
        # Fungsi `get_tensor()` mengembalikan salinan data tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_label_index = np.argmax(output_data[0])
        predicted_label = labels[predicted_label_index]

        # Ambil data nutrisi berdasarkan label prediksi
        nutrition_data, breakdown_details = scrape_nutrition_data(predicted_label)

        return render(request, 'dashboard/predict_image.html', {
            'label': predicted_label,
            'image_url': image_url,
            'nutrition_data': nutrition_data,
            'breakdown_details': breakdown_details
        })

    return render(request, 'dashboard/predict_image.html')
