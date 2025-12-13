from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from PIL import Image
from fastapi.responses import JSONResponse
from pipeline.recognise import recognise
import numpy as np
import uvicorn
import cv2
from mtcnn import MTCNN
from pipeline.recognise import recognise
from sklearn.metrics.pairwise import cosine_similarity
from src.constants import EMBEDDINGS_PATH


app = FastAPI(title="Face Recognition API (prototype)")

@app.get("/")
async def greet():
    return { "message": "Your API is up and running"}

# @app.post("/register")
# async def register(name: str = Form(...), file: UploadFile = File(...)):
#     """Register a face with label 'name'. Multiple faces in image will each be added."""
#     content = await file.read()
#     encs = get_embeddings_from_image_bytes(content)
#     if not encs:
#         raise HTTPException(status_code=400, detail="No face found in image")
#     # Add to db
#     global db
#     if name not in db:
#         db[name] = []
#     for e in encs:
#         db[name].append(e)
#     save_db(db)
#     return {"status": "ok", "name": name, "added": len(encs)}

# @app.post("/identify")
# async def identify(file: UploadFile = File(...), top_k: int = 3):
#     """Identify faces in the uploaded image. Returns best matches per face."""
#     content = await file.read()
#     encs = get_embeddings_from_image_bytes(content)
#     if not encs:
#         raise HTTPException(status_code=400, detail="No face found in image")
#     # Build embedding matrix & labels
#     labels = []
#     embs = []
#     for label, enc_list in db.items():
#         for e in enc_list:
#             labels.append(label)
#             embs.append(e)
#     if not embs:
#         raise HTTPException(status_code=404, detail="No enrolled faces in DB")
#     embs = np.vstack(embs)  # (N, 128)
#     results = []
#     for face_enc in encs:
#         sims = cosine_similarity(face_enc.reshape(1, -1), embs).flatten()  # cosine in [-1,1]
#         # get top_k indices
#         idxs = np.argsort(-sims)[:top_k]
#         hits = [{"label": labels[i], "score": float(sims[i])} for i in idxs]
#         results.append({"matches": hits})
#     return {"faces_found": len(encs), "results": results}

@app.post("/verify")
async def verify(name: str = Form(...), file: UploadFile = File(...), threshold: float = 0.2):
    """Verify whether uploaded image matches given 'name'."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="No image detected")
    np_arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    detector = MTCNN()
    faces = detector.detect_faces(image = img)
    if len(faces) == 0:
        raise HTTPException(status_code=400, detail=f"No face detected")
    person, max_similarity = recognise(img_array = img)
    match = person.lower() == name.lower()
    return {"name": name, "match": bool(match)}

@app.get("/health")
def health():
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    if data is None:
        raise HTTPException(status_code = 404, detail = "Data not found...")
    
    names = data["names"]
    return {"status": "ok", "registered": list(names)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
