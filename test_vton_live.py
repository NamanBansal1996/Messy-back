from gradio_client import Client, handle_file
import urllib.request
import traceback

urllib.request.urlretrieve('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png', 'person.png')
urllib.request.urlretrieve('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png', 'garment.png')

client = Client('yisol/IDM-VTON')

try:
    result = client.predict(
        {"background": handle_file('person.png'), "layers": [], "composite": None},
        handle_file('garment.png'),
        "clothing item",
        True,
        False,
        30,
        42,
        api_name="/tryon"
    )
    print("SUCCESS", result)
except Exception as e:
    print("FAILED", e)
    traceback.print_exc()
