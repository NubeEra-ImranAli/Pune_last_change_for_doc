from django.urls import path,include
from django.contrib import admin
from qualityindex import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
   
    path('admin/', admin.site.urls),

    path('',views.predict_aqi,name=''),
    path('home',views.predict_aqi,name='home'),
    path('charts',views.charts,name='charts'),
    path('user_result',views.user_result_view,name='user_result'),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
