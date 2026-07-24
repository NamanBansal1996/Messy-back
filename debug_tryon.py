import sys
import os

sys.path.append("/Users/namanbansal/Projects/messy/messy-back")
from virtual_tryon import try_on_with_idmvton, file_to_base64

person_path = "/Users/namanbansal/Projects/Messy/Messy-front/public/teenage-girl-slouching-on-chair-photo.jpg"
garment_path = "/Users/namanbansal/Projects/Messy/Messy-front/public/menlooset-shirtlevis.webp"

person_b64 = file_to_base64(person_path)
garment_b64 = file_to_base64(garment_path)

print("Starting TryOn Test...")
res, err = try_on_with_idmvton(person_b64, garment_b64, "upper")
print("RESULT SUCCESS:", res is not None)
if err:
    print("ERROR:", err)
