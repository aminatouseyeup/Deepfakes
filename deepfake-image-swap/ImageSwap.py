import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model(
    "deepfake-image-swap/inswapper_128.onnx", download=False, download_zip=False
)


def detect_faces(img):
    faces = app.get(img)

    return faces


def swap_all(img, faces, face_num):
    res = img.copy()
    source_face = faces[face_num]

    for face in faces:
        res = swapper.get(res, face, source_face, paste_back=True)

    return res


def swap_one(img, face1, face2):
    res = img.copy()
    res = swapper.get(res, face1, face2, paste_back=True)

    return res
