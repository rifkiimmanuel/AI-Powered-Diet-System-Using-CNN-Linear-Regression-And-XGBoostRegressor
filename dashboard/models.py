from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

# Create your models here.
# Define categories as a list of tuples
CATEGORY = [
    ('Dessert', 'Dessert'),
    ('Lunch', 'Lunch'),
    ('Dinner', 'Dinner'),
    ('Breakfast', 'Breakfast'),
    ('Fruit', 'Fruit'),
    ('Appetizer', 'Appetizer'),
    ('Drink', 'Drink'),
    ('Juice', 'Juice'),
    ('Coffee', 'Coffee'),
    ('Milk', 'Milk'),
    ('Tea', 'Tea'),
]


class Calories(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='calories', null = True)
    food_name = models.CharField(max_length = 255, null = True)
    calories_amount = models.PositiveIntegerField(null = True)
    serving = models.PositiveIntegerField(null = True)
    category = models.CharField(max_length = 255, choices= CATEGORY, null = True)
    date = models.DateTimeField(default=timezone.now)

    class Meta:
        verbose_name_plural = 'Calories'

    def __str__(self):
        return f'{self.food_name}-{self.calories_amount}'
