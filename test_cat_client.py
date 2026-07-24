from gradio_client import Client, handle_file
import urllib.request
import os

print("Downloading test images...")
urllib.request.urlretrieve('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png', 'person.png')
urllib.request.urlretrieve('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png', 'garment.png')

print("Connecting to CatVTON...")
try:
    client = Client("zhengchong/CatVTON")
    res = client.predict(
        person_image={"background": handle_file("person.png"), "layers": [], "composite": None},
        cloth_image=handle_file("garment.png"),
        num_inference_steps=50,
        guidance_scale=2.5,
        seed=42,
        api_name="/submit_function_p2p"
    )
    print("SUCCESS!", res)
except Exception as e:
    print("FAILED!", e)

