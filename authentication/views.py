from django.shortcuts import render, redirect

from django.contrib.auth.forms import UserCreationForm

from .forms import CreateUserForm, UserUpdateForm, ProfileUpdateForm
from django.contrib import messages
from django.contrib.auth.views import LoginView
from django.contrib import messages
from django.contrib.auth.decorators import login_required

def register(request):
    form = CreateUserForm()

    if request.method  == 'POST':
        form = CreateUserForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'{username} account has been created')
            return redirect('user-login')
        
    else:
        form = CreateUserForm()

    context = {'form':form}
    return render(request, 'user/register.html', context)



from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
    

def profile(request):
    
    return render(request, 'user/profile.html')


from django.shortcuts import render
from django.contrib.auth.models import User



def profile_update(request):
    if request.method == 'POST':
        user_form = UserUpdateForm(request.POST, instance=request.user)
        profile_form = ProfileUpdateForm(request.POST, request.FILES, instance=request.user.profile)
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            return redirect('user-profile')
    else: 
        user_form = UserUpdateForm(instance=request.user)
        profile_form = ProfileUpdateForm(instance=request.user.profile)

    context = {
        'user_form':user_form,
        'profile_form':profile_form,

    }
    return render(request, 'user/profile_update.html', context)


@login_required
def delete_account(request):
    if request.method == 'POST':
        user = request.user
        user.delete()
        messages.success(request, 'Your account has been deleted successfully.')
        return redirect('user-register')  # Redirect to home or login page after deletion
    return render(request, 'user/delete_account_confirm.html')