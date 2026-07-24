import requests

url = "http://localhost:5000/tryon"
files = {
    "person_image": ("person.jpg", open("/Users/namanbansal/Projects/Messy/Messy-front/public/teenage-girl-slouching-on-chair-photo.jpg", "rb")),
    "garment_image": ("garment.webp", open("/Users/namanbansal/Projects/Messy/Messy-front/public/menlooset-shirtlevis.webp", "rb"))
}
data = {"garment_type": "upper"}

print("Sending POST to", url)
res = requests.post(url, files=files, data=data)
print("Status:", res.status_code)
print("Response:", res.json() if res.status_code == 503 or res.status_code == 200 else res.text)
