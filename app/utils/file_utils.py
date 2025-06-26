# IN this we are doing naming ot hte uploaded file and also delete them if any prefivous file exis  with the same name
import os
import uuid

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_upload_file(file):
    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return file_path

def delete_file(path):
    if os.path.exists(path):
        os.remove(path)
