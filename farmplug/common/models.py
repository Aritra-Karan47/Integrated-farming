# common/models.py
from django.db import models
from django.contrib.auth.models import User
from django_tenants.models import TenantMixin, DomainMixin

class Client(TenantMixin):
    name = models.CharField(max_length=100)
    paid_until = models.DateField()
    on_trial = models.BooleanField()
    created_on = models.DateField(auto_now_add=True)

    auto_create_schema = True

class Domain(DomainMixin):
    pass

class FarmingType(models.Model):
    name = models.CharField(max_length=50, unique=True)
    tasks = models.TextField()  # JSON or text for daily tasks

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    client = models.ForeignKey(Client, on_delete=models.CASCADE)
    farming_types = models.ManyToManyField(FarmingType)

# Signal to create profile and schema on user creation
from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver(post_save, sender=User)
def create_profile(sender, instance, created, **kwargs):
    if created:
        # Create tenant schema for new user
        tenant = Client(name=instance.username, schema_name=instance.username.lower().replace(' ', '_'), paid_until='2025-12-31', on_trial=True)
        tenant.save()
        domain = Domain(domain=f"{tenant.schema_name}.localhost", tenant=tenant, is_primary=True)
        domain.save()
        UserProfile.objects.create(user=instance, client=tenant)