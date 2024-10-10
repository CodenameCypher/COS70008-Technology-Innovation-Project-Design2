from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
import pandas as pd
import io
import csv
import base64
import matplotlib.pyplot as plt
from .forms import UploadFileForm  # Ensure you have this form defined
from .models import UploadedFile, Malem  # Ensure you have UploadedFile and Malem models defined
from django.conf import settings
import os
import shutil


from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.contrib.auth import get_user_model

from django.contrib.sessions.models import Session
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from django.shortcuts import render

from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.shortcuts import render


from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.contrib.sessions.models import Session
from django.utils import timezone


from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.contrib.sessions.models import Session
from django.utils import timezone
from django.shortcuts import render

@login_required
def user_management(request):
    User = get_user_model()
    users = User.objects.all()  # Fetch all registered users

    # Retrieve currently active users
    active_sessions = Session.objects.filter(expire_date__gte=timezone.now())
    logged_in_users = []
    for session in active_sessions:
        session_data = session.get_decoded()
        user_id = session_data.get('_auth_user_id')
        if user_id:
            try:
                user = User.objects.get(id=user_id)
                logged_in_users.append(user)
            except User.DoesNotExist:
                continue  # The user may have been deleted

    return render(request, 'employee_information/user.html', {
        'users': users,
        'logged_in_users': logged_in_users,  # Pass the actual user objects
    })


UPLOAD_DIR = os.path.join(settings.MEDIA_ROOT, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

from django.contrib.auth import get_user_model

# Registration View
def register_user(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']
        
        # Ensure email uniqueness
        if User.objects.filter(email=email).exists():
            messages.error(request, "This email is already registered.")
        else:
            try:
                # Set email as both username and email for the user
                user = get_user_model().objects.create_user(username=email, email=email, password=password)
                messages.success(request, "Registration successful! You can now log in.")
                return redirect('login')
            except Exception as e:
                messages.error(request, f"Registration failed: {str(e)}")
    
    return render(request, 'employee_information/register.html')

# Login View
from .models import UserLog

# Login View
def login_user(request):
    logout(request)
    resp = {"status": 'failed', 'msg': ''}
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']

        user = authenticate(username=email, password=password)
        if user is not None:
            if user.is_active:
                login(request, user)
                return redirect('home')
            else:
                resp['msg'] = "Account is inactive."
        else:
            resp['msg'] = "Incorrect email or password."
    
    return render(request, 'employee_information/login.html', {'resp': resp})




@login_required
def user_log_graph(request):
    logs = UserLog.objects.all()
    timestamps = [log.timestamp for log in logs]
    counts = [logs.filter(timestamp__date=log.timestamp.date()).count() for log in logs]

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, counts, marker='o')
    plt.title('User Log Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Logs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot to a BytesIO object and encode to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    graph = base64.b64encode(buf.getvalue()).decode('utf-8')
    return render(request, 'employee_information/admin_dashboard.html', {'graph': graph})


# Logout View
def logoutuser(request):
    logout(request)
    return redirect('/')

# Home View
@login_required
def home(request):
    return render(request, 'employee_information/home.html', {'page_title': 'Home'})

# About View
def about(request):
    return render(request, 'employee_information/about.html', {'page_title': 'About'})

# Function to generate a pie chart based on severity counts
def generate_pie_chart(dataframe):
    severity_counts = dataframe['severity'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%', colors=['blue', 'grey'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Convert plot to PNG image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')





UPLOAD_DIR = os.path.join(settings.MEDIA_ROOT, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)




# views.py
from django.shortcuts import render
from .models import FileInfo  # Update with the actual model name for file records

@login_required
def dashboard(request):
    file_info_list = FileInfo.objects.all()  # Fetch all records from the FileInfo model

    # Get data for the pie chart
    malem_data = Malem.objects.all()  # Fetch all records from Malem model

    # Convert Malem data to a DataFrame for easier analysis
    df = pd.DataFrame(list(malem_data.values('category')))  # Replace 'category' with the relevant field

    # Generate the pie chart
    chart_image = generate_pie_chart(df)

    return render(request, 'employee_information/dashboard.html', {
        'file_info_list': file_info_list,
        'chart_image': chart_image  # Pass the chart image to the template
    })




UPLOAD_DIR = os.path.join(settings.MEDIA_ROOT, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

def handle_uploaded_file(file, request):
    file_path = os.path.join(UPLOAD_DIR, file.name)
    with open(file_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    return file_path

@login_required
def upload_file(request):
    file_info_list = FileInfo.objects.all()  # Retrieve all file records to display in the table

    if request.method == 'POST':
        files = request.FILES.getlist('file')  # Get all uploaded files
        
        for file in files:
            # Save the file and handle disk space
            file_path = handle_uploaded_file(file)

            # Save file info to the database
            FileInfo.objects.create(
                name=file.name,
                size=file.size,
                progress="Uploaded"
            )
        
        messages.success(request, "File uploaded successfully.")
        return redirect('upload_file')  # Redirect to refresh the page

    return render(request, 'upload_file.html', {'file_info_list': file_info_list})

@login_required
def analysis(request):
    file_info_list = []  # Initialize a list to hold file information

    if request.method == 'POST':
        files = request.FILES.getlist('file')  # Get a list of uploaded files

        for file in files:
            # Save the file to a temporary location
            file_path = handle_uploaded_file(file, request)

            if file_path:
                # Save file info to the database
                file_info = FileInfo.objects.create(
                    name=file.name,
                    size=file.size,
                    progress="Uploaded"  # Set progress as 'Uploaded'
                )

                file_info_list.append({
                    'name': file_info.name,
                    'size': file_info.size,
                    'progress': file_info.progress,
                })

                # Check if the file name contains 'malem'
                if 'malem' in file.name.lower():
                    # Use pandas to read the CSV and limit to 2000 rows
                    df = pd.read_csv(file_path, nrows=2000)

                    # Check if 'Category' exists in the columns
                    if 'Category' in df.columns:
                        malem_objects = []  # List to hold Malem objects for bulk creation

                        for index, row in df.iterrows():
                            malem_objects.append(Malem(
                                category=row.get('Category', ''),
                                pslist_nproc=int(row.get('pslist.nproc', 0)),
                                pslist_nppid=int(row.get('pslist.nppid', 0)),
                                pslist_avg_threads=float(row.get('pslist.avg_threads', 0)),
                                pslist_nprocs64bit=int(row.get('pslist.nprocs64bit', 0)),
                                pslist_avg_handlers=float(row.get('pslist.avg_handlers', 0)),
                                dlllist_ndlls=int(row.get('dlllist.ndlls', 0)),
                                dlllist_avg_dlls_per_proc=float(row.get('dlllist.avg_dlls_per_proc', 0)),
                                handles_nhandles=int(row.get('handles.nhandles', 0)),
                                handles_avg_handles_per_proc=float(row.get('handles.avg_handles_per_proc', 0)),
                                handles_nport=int(row.get('handles.nport', 0)),
                                handles_nfile=int(row.get('handles.nfile', 0)),
                                handles_nevent=int(row.get('handles.nevent', 0)),
                                handles_ndesktop=int(row.get('handles.ndesktop', 0)),
                                handles_nkey=int(row.get('handles.nkey', 0)),
                                handles_nthread=int(row.get('handles.nthread', 0)),
                                handles_ndirectory=int(row.get('handles.ndirectory', 0)),
                                handles_nsemaphore=int(row.get('handles.nsemaphore', 0)),
                                handles_ntimer=int(row.get('handles.ntimer', 0)),
                                handles_nsection=int(row.get('handles.nsection', 0)),
                                handles_nmutant=int(row.get('handles.nmutant', 0)),
                                ldrmodules_not_in_load=int(row.get('ldrmodules.not_in_load', 0)),
                                ldrmodules_not_in_init=int(row.get('ldrmodules.not_in_init', 0)),
                                ldrmodules_not_in_mem=int(row.get('ldrmodules.not_in_mem', 0)),
                                ldrmodules_not_in_load_avg=float(row.get('ldrmodules.not_in_load_avg', 0)),
                                ldrmodules_not_in_init_avg=float(row.get('ldrmodules.not_in_init_avg', 0)),
                                ldrmodules_not_in_mem_avg=float(row.get('ldrmodules.not_in_mem_avg', 0)),
                                malfind_ninjections=int(row.get('malfind.ninjections', 0)),
                                malfind_commitCharge=float(row.get('malfind.commitCharge', 0)),
                                malfind_protection=float(row.get('malfind.protection', 0)),
                                malfind_uniqueInjections=int(row.get('malfind.uniqueInjections', 0)),
                                psxview_not_in_pslist=int(row.get('psxview.not_in_pslist', 0)),
                                psxview_not_in_eprocess_pool=int(row.get('psxview.not_in_eprocess_pool', 0)),
                                psxview_not_in_ethread_pool=int(row.get('psxview.not_in_ethread_pool', 0)),
                                psxview_not_in_pspcid_list=int(row.get('psxview.not_in_pspcid_list', 0)),
                                psxview_not_in_csrss_handles=int(row.get('psxview.not_in_csrss_handles', 0)),
                                psxview_not_in_session=int(row.get('psxview.not_in_session', 0)),
                                psxview_not_in_deskthrd=int(row.get('psxview.not_in_deskthrd', 0)),
                                psxview_not_in_pslist_false_avg=float(row.get('psxview.not_in_pslist_false_avg', 0)),
                                psxview_not_in_eprocess_pool_false_avg=float(row.get('psxview.not_in_eprocess_pool_false_avg', 0)),
                                psxview_not_in_ethread_pool_false_avg=float(row.get('psxview.not_in_ethread_pool_false_avg', 0)),
                                psxview_not_in_pspcid_list_false_avg=float(row.get('psxview.not_in_pspcid_list_false_avg', 0)),
                                psxview_not_in_csrss_handles_false_avg=float(row.get('psxview.not_in_csrss_handles_false_avg', 0)),
                                psxview_not_in_session_false_avg=float(row.get('psxview.not_in_session_false_avg', 0)),
                                psxview_not_in_deskthrd_false_avg=float(row.get('psxview.not_in_deskthrd_false_avg', 0)),
                                modules_nmodules=int(row.get('modules.nmodules', 0)),
                                svcscan_nservices=int(row.get('svcscan.nservices', 0)),
                                svcscan_kernel_drivers=int(row.get('svcscan.kernel_drivers', 0)),
                                svcscan_fs_drivers=int(row.get('svcscan.fs_drivers', 0)),
                                svcscan_process_services=int(row.get('svcscan.process_services', 0)),
                                svcscan_shared_process_services=int(row.get('svcscan.shared_process_services', 0)),
                                svcscan_interactive_process_services=int(row.get('svcscan.interactive_process_services', 0)),
                                svcscan_nactive=int(row.get('svcscan.nactive', 0)),
                                callbacks_ncallbacks=int(row.get('callbacks.ncallbacks', 0)),
                                callbacks_nanonymous=int(row.get('callbacks.nanonymous', 0)),
                                callbacks_ngeneric=int(row.get('callbacks.ngeneric', 0)),
                                class_name=row.get('Class', ''),
                            ))

# Bulk create Malem objects
                        Malem.objects.bulk_create(malem_objects)
                    else:
                        messages.error(request, "The uploaded file is missing the 'Category' column.")
                        return render(request, 'employee_information/analysis.html', {'file_info_list': file_info_list})

        messages.success(request, "Files uploaded and data processed successfully.")
        return render(request, 'employee_information/analysis.html', {'file_info_list': file_info_list})

    return render(request, 'employee_information/analysis.html', {'file_info_list': file_info_list})





from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
import pandas as pd
import io
import csv
import base64
import matplotlib.pyplot as plt
from .forms import UploadFileForm  # Ensure you have this form defined
from .models import UploadedFile, Malem  # Ensure you have UploadedFile and Malem models defined
from django.conf import settings
import os
import shutil
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

UPLOAD_DIR = os.path.join(settings.MEDIA_ROOT, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Other views remain unchanged...

def train_knn_model(dataframe):
    """ Train a KNN model based on the provided dataframe. """
    features = dataframe.drop(columns=['Category'])  # Assuming 'Category' is the target variable
    labels = dataframe['Category']

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features, labels)
    
    return knn

def predict_ransomware(model, data):
    """ Predict whether the given data points exhibit ransomware characteristics. """
    predictions = model.predict(data)
    return predictions


from django.core.files.storage import FileSystemStorage
from io import BytesIO
import base64

def generate_pie_chart(dataframe):
    # Assuming the column you want to visualize is named 'category'
    severity_counts = dataframe['category'].value_counts()  # Change 'severity' to 'category'
    fig, ax = plt.subplots()
    ax.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%', colors=plt.cm.tab20.colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Convert plot to PNG image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')





# views.py

import json

from django.shortcuts import render
import json
from .models import Malem  # Ensure Malem is correctly imported

import json
import numpy as np
from django.shortcuts import render
from sklearn.neighbors import KNeighborsClassifier
from .models import Malem  # Make sure to import your Malem model

import numpy as np
import json
from django.shortcuts import render
from sklearn.neighbors import KNeighborsClassifier
from .models import Malem  # Adjust import as per your project structure

import numpy as np
import json
from django.shortcuts import render
from sklearn.neighbors import KNeighborsClassifier
from .models import Malem  # Ensure you import your model

def calculate_severity(issue):
    """
    Calculate the severity level based on the features of the issue.
    This is a simple example; adjust the thresholds as necessary.
    """
    # Example thresholds for severity levels
    if issue.malfind_ninjections > 10:
        return "high"
    elif issue.malfind_ninjections > 5:
        return "medium"
    else:
        return "low"

def analysis_visualization(request):
    # Fetch all instances of the Malem model
    issues_data = Malem.objects.all()

    # Prepare data for KNN and calculations
    feature_list = []
    severity_list = []
    
    for issue in issues_data:
        # Collect feature values
        features = [
            issue.pslist_nproc,
            issue.pslist_nppid,
            issue.pslist_avg_threads,
            issue.pslist_nprocs64bit,
            issue.pslist_avg_handlers,
            issue.dlllist_ndlls,
            issue.dlllist_avg_dlls_per_proc,
            issue.handles_nhandles,
            issue.handles_avg_handles_per_proc,
            issue.handles_nport,
            issue.handles_nfile,
            issue.handles_nevent,
            issue.handles_ndesktop,
            issue.handles_nkey,
            issue.handles_nthread,
            issue.handles_ndirectory,
            issue.handles_nsemaphore,
            issue.handles_ntimer,
            issue.handles_nsection,
            issue.handles_nmutant,
            issue.ldrmodules_not_in_load,
            issue.ldrmodules_not_in_init,
            issue.ldrmodules_not_in_mem,
            issue.malfind_ninjections,
            issue.malfind_commitCharge,
            issue.malfind_protection,
            issue.malfind_uniqueInjections,
            issue.psxview_not_in_pslist,
            issue.psxview_not_in_eprocess_pool,
            issue.psxview_not_in_ethread_pool,
            issue.psxview_not_in_pspcid_list,
            issue.psxview_not_in_csrss_handles,
            issue.psxview_not_in_session,
            issue.psxview_not_in_deskthrd,
            issue.modules_nmodules,
            issue.svcscan_nservices,
            issue.svcscan_kernel_drivers,
            issue.svcscan_fs_drivers,
            issue.svcscan_process_services,
            issue.svcscan_shared_process_services,
            issue.svcscan_interactive_process_services,
            issue.svcscan_nactive,
            issue.callbacks_ncallbacks,
            issue.callbacks_nanonymous,
            issue.callbacks_ngeneric
        ]
        
        feature_list.append(features)

        # Calculate and append severity for each issue
        severity = calculate_severity(issue)
        severity_list.append(severity)

    # Convert to numpy arrays for KNN
    X = np.array(feature_list)
    y = np.array(severity_list)

    # Ensure that all features are numeric
    try:
        X = np.asarray(X, dtype=np.float64)  # Convert to float64, will raise ValueError if not possible
    except ValueError as e:
        print("Error converting features to numeric:", e)
        # Handle the exception appropriately, e.g., return an error message

    # Ensure severity is numeric (you might need to map severity levels to integers)
    unique_severities = list(set(severity_list))
    severity_mapping = {severity: idx for idx, severity in enumerate(unique_severities)}
    y = np.array([severity_mapping[severity] for severity in severity_list])

    # Use KNN for classification - Example: Using k=3
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    # Predict and count ransomware occurrences
    predictions = knn.predict(X)
    ransomware_count = np.sum(predictions == severity_mapping.get("ransomware", -1))  # Count ransomware predictions

    # Calculate total detected issues (sum of individual counts)
    total_detected_issues = len(issues_data)  # Each instance is considered an issue

    # Severity distribution
    severity_counts = {}
    for severity in severity_list:
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    severity_labels = list(severity_counts.keys())
    severity_values = list(severity_counts.values())

    # Class distribution (using class_name field)
    class_counts = {}
    for issue in issues_data:
        issue_class = issue.class_name
        class_counts[issue_class] = class_counts.get(issue_class, 0) + 1

    class_labels = list(class_counts.keys())
    class_values = list(class_counts.values())

    context = {
        'total_detected_issues': total_detected_issues,
        'ransomware_count': ransomware_count,
        'severity_labels': json.dumps(severity_labels),
        'severity_values': json.dumps(severity_values),
        'class_labels': json.dumps(class_labels),
        'class_values': json.dumps(class_values),
    }

    return render(request, 'employee_information/file_analysis.html', context)



# views.py
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from .models import CustomUser

@login_required
def admin_dashboard(request):
    # Redirect if the user is not an admin
    if not request.user.is_admin:
        return redirect('user_dashboard')

    # Fetch all instances of the Malem model
    issues_data = Malem.objects.all()

    # Prepare data for KNN and calculations
    feature_list = []
    severity_list = []

    for issue in issues_data:
        features = [
            issue.pslist_nproc,
            issue.pslist_nppid,
            issue.pslist_avg_threads,
            issue.pslist_nprocs64bit,
            issue.pslist_avg_handlers,
            issue.dlllist_ndlls,
            issue.dlllist_avg_dlls_per_proc,
            issue.handles_nhandles,
            issue.handles_avg_handles_per_proc,
            issue.handles_nport,
            issue.handles_nfile,
            issue.handles_nevent,
            issue.handles_ndesktop,
            issue.handles_nkey,
            issue.handles_nthread,
            issue.handles_ndirectory,
            issue.handles_nsemaphore,
            issue.handles_ntimer,
            issue.handles_nsection,
            issue.handles_nmutant,
            issue.ldrmodules_not_in_load,
            issue.ldrmodules_not_in_init,
            issue.ldrmodules_not_in_mem,
            issue.malfind_ninjections,
            issue.malfind_commitCharge,
            issue.malfind_protection,
            issue.malfind_uniqueInjections,
            issue.psxview_not_in_pslist,
            issue.psxview_not_in_eprocess_pool,
            issue.psxview_not_in_ethread_pool,
            issue.psxview_not_in_pspcid_list,
            issue.psxview_not_in_csrss_handles,
            issue.psxview_not_in_session,
            issue.psxview_not_in_deskthrd,
            issue.modules_nmodules,
            issue.svcscan_nservices,
            issue.svcscan_kernel_drivers,
            issue.svcscan_fs_drivers,
            issue.svcscan_process_services,
            issue.svcscan_shared_process_services,
            issue.svcscan_interactive_process_services,
            issue.svcscan_nactive,
            issue.callbacks_ncallbacks,
            issue.callbacks_nanonymous,
            issue.callbacks_ngeneric
        ]
        
        feature_list.append(features)

        # Calculate and append severity for each issue
        severity = calculate_severity(issue)
        severity_list.append(severity)

    # Convert to numpy arrays for KNN
    X = np.array(feature_list)
    y = np.array(severity_list)

    # Ensure that all features are numeric
    try:
        X = np.asarray(X, dtype=np.float64)  # Convert to float64
    except ValueError as e:
        print("Error converting features to numeric:", e)
        return render(request, 'admin_dashboard.html', {'error': str(e)})

    # Create severity mapping and fit KNN
    unique_severities = list(set(severity_list))
    severity_mapping = {severity: idx for idx, severity in enumerate(unique_severities)}
    y = np.array([severity_mapping[severity] for severity in severity_list])

    # Use KNN for classification - Example: Using k=3
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    # Predict and count severity
    predictions = knn.predict(X)
    severity_counts = {severity: 0 for severity in unique_severities}
    for prediction in predictions:
        severity_counts[unique_severities[prediction]] += 1

    # Prepare context for the dashboard
    context = {
        'total_detected_issues': len(issues_data),
        'severity_counts': severity_counts,
        'severity_labels': json.dumps(list(severity_counts.keys())),
        'severity_values': json.dumps(list(severity_counts.values())),
    }

    return render(request, 'employee_information/admin_dashboard.html', context)



@login_required
def manage_users(request):
    if not request.user.is_admin:
        return redirect('user_dashboard')
    users = CustomUser.objects.all()
    return render(request, 'employee_information/manage_users.html', {'users': users})

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import CustomUser  # Adjust the import based on your actual model

@login_required
def edit_user(request, user_id):
    # Get the user object or return a 404 if not found
    user = get_object_or_404(CustomUser, id=user_id)

    if request.method == 'POST':
        # Update user details from the request
        user.username = request.POST.get('username')
        user.email = request.POST.get('email')
        # Add other fields as necessary

        user.save()  # Save the updated user object
        return redirect('user_management')  # Redirect to the user management page

    return render(request, 'employee_information/edit_user.html', {'user': user})  # Render the form for GET requests

@login_required
def delete_user(request, user_id):
    if not request.user.is_admin:
        return redirect('user_dashboard')
    user = CustomUser.objects.get(id=user_id)
    user.delete()
    return redirect('manage_users')




