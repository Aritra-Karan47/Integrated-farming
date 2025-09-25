# farmplug/urls.py
from django.contrib import admin
from django.urls import path, include
from django_tenants.urlresolvers import TenantURLResolver

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('common.urls')),  # Login/profile
]

tenant_urlresolver = TenantURLResolver()
urlpatterns += tenant_urlresolver.get_urlpatterns()