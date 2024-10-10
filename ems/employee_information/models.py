from django.db import models
from django.utils import timezone
from django.conf import settings
from django.contrib.auth.models import AbstractUser  # Make sure to import AbstractUser


from django.contrib.auth import get_user_model

User = get_user_model()



class UploadedFile(models.Model):
    name = models.CharField(max_length=255)
    upload = models.FileField(upload_to='uploads/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class FileInfo(models.Model):
    name = models.CharField(max_length=255)
    size = models.PositiveIntegerField()  # Size in bytes
    progress = models.CharField(max_length=50)
    upload_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class Malem(models.Model):
    category = models.CharField(max_length=50)
    pslist_nproc = models.IntegerField()
    pslist_nppid = models.IntegerField()
    pslist_avg_threads = models.FloatField()
    pslist_nprocs64bit = models.IntegerField()
    pslist_avg_handlers = models.FloatField()
    dlllist_ndlls = models.IntegerField()
    dlllist_avg_dlls_per_proc = models.FloatField()
    handles_nhandles = models.IntegerField()
    handles_avg_handles_per_proc = models.FloatField()
    handles_nport = models.IntegerField()
    handles_nfile = models.IntegerField()
    handles_nevent = models.IntegerField()
    handles_ndesktop = models.IntegerField()
    handles_nkey = models.IntegerField()
    handles_nthread = models.IntegerField()
    handles_ndirectory = models.IntegerField()
    handles_nsemaphore = models.IntegerField()
    handles_ntimer = models.IntegerField()
    handles_nsection = models.IntegerField()
    handles_nmutant = models.IntegerField()
    ldrmodules_not_in_load = models.IntegerField()
    ldrmodules_not_in_init = models.IntegerField()
    ldrmodules_not_in_mem = models.IntegerField()
    ldrmodules_not_in_load_avg = models.FloatField()
    ldrmodules_not_in_init_avg = models.FloatField()
    ldrmodules_not_in_mem_avg = models.FloatField()
    malfind_ninjections = models.IntegerField()
    malfind_commitCharge = models.FloatField()
    malfind_protection = models.FloatField()
    malfind_uniqueInjections = models.IntegerField()
    psxview_not_in_pslist = models.IntegerField()
    psxview_not_in_eprocess_pool = models.IntegerField()
    psxview_not_in_ethread_pool = models.IntegerField()
    psxview_not_in_pspcid_list = models.IntegerField()
    psxview_not_in_csrss_handles = models.IntegerField()
    psxview_not_in_session = models.IntegerField()
    psxview_not_in_deskthrd = models.IntegerField()
    psxview_not_in_pslist_false_avg = models.FloatField()
    psxview_not_in_eprocess_pool_false_avg = models.FloatField()
    psxview_not_in_ethread_pool_false_avg = models.FloatField()
    psxview_not_in_pspcid_list_false_avg = models.FloatField()
    psxview_not_in_csrss_handles_false_avg = models.FloatField()
    psxview_not_in_session_false_avg = models.FloatField()
    psxview_not_in_deskthrd_false_avg = models.FloatField()
    modules_nmodules = models.IntegerField()
    svcscan_nservices = models.IntegerField()
    svcscan_kernel_drivers = models.IntegerField()
    svcscan_fs_drivers = models.IntegerField()
    svcscan_process_services = models.IntegerField()
    svcscan_shared_process_services = models.IntegerField()
    svcscan_interactive_process_services = models.IntegerField()
    svcscan_nactive = models.IntegerField()
    callbacks_ncallbacks = models.IntegerField()
    callbacks_nanonymous = models.IntegerField()
    callbacks_ngeneric = models.IntegerField()
    class_name = models.CharField(max_length=50)
    handles_nport = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.category} - {self.class_name}"


class UserLog(models.Model):
    user = models.ForeignKey('employee_information.CustomUser', on_delete=models.CASCADE)  # Explicit reference
    action = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.action} at {self.timestamp}"




from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    # Add related_name to avoid reverse accessor clashes
    groups = models.ManyToManyField(
        'auth.Group',
        related_name='customuser_set',  # Avoid clashes with auth.User.groups
        blank=True,
        help_text='The groups this user belongs to.',
        verbose_name='groups'
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='customuser_set',  # Avoid clashes with auth.User.user_permissions
        blank=True,
        help_text='Specific permissions for this user.',
        verbose_name='user permissions'
    )

