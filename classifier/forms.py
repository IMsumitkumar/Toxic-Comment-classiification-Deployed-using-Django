from django import forms


class InputForm(forms.Form):
    comment = forms.CharField(max_length=500)
