from virtual_tryon import generate_tryon
import base64
import os

with open("test_person.jpg", "wb") as f:
    f.write(os.urandom(1024))
with open("test_garm.jpg", "wb") as f:
    f.write(os.urandom(1024))

person_b64 = base64.b64encode(os.urandom(1024)).decode("utf-8")
garm_b64 = base64.b64encode(os.urandom(1024)).decode("utf-8")

res = generate_tryon(person_b64, garm_b64, "upper")
print(res)
