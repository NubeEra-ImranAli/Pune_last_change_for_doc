from django import template

register = template.Library()

@register.filter
def get_aqi_color(category):
    colors = {
        "Good": "Green",
        "Moderate": "Yellow",
        "Unhealthy for Sensitive Groups": "Orange",
        "Unhealthy": "Red",
        "Very Unhealthy": "Purple",
        "Hazardous": "Maroon",
    }
    return colors.get(category, "#ffffff")  # Default to white if unknown