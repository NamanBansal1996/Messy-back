from gradio_client import Client, handle_file

try:
    client = Client("zhengchong/CatVTON")
    client.view_api()
except Exception as e:
    print(f"Error: {e}")

