import streamlit as st
import numpy as np
import os
import sys
import time
import threading

sys.path.append("deepfake-image-detector")
sys.path.append("deepfake-image-swap")
sys.path.append("deepfake-audio-generator")
sys.path.append("deepfake-audio-generator/Real-Time-Voice-Cloning")
sys.path.append("deepfake-video-generator")

# from mesoModel import predict_image
from ImageSwap import detect_faces, swap_all, swap_one

# from AudioGenerator import generate_audio
from videogenerator import deep_fake_animation


import cv2
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# CSS for the logo
st.markdown("""<style>
    .title {
        font-size: 60px;
        font-weight: bold;
        margin: 0;
        align-items: center;
    }
    .logo {
        display: flex;
        align-items: center;
        justify-content: center;}</style>""", unsafe_allow_html=True)

# Logo
col1, col2 = st.columns([1, 9])
with col1:
    st.image('imageApp/logo.png', width=100)
with col2:
    st.markdown('<div class="title">DeepFake Studio</div>', unsafe_allow_html=True)

# Background color
def set_bg_color():
    st.markdown("""
        <style>
        .stApp {
            background-color: #E9EDF0; 
        }
        </style>
        """, unsafe_allow_html=True)
set_bg_color()

#Home page
def home():
    st.title("Welcome to DeepFake Studio")

    st.markdown("""
        🌐 **Discover the fascinating world of deepfakes** 🌐

        In a world where reality and fiction often intertwine, deepfakes have emerged as a revolutionary technology capable of creating content that bears an astounding resemblance to the real thing. 🎭 Derived from artificial intelligence and deep learning techniques, deepfakes are videos, images or audio recordings manipulated to make a person say or do something who never actually said or did it. The technology takes its name from “deep learning” and “fake” and is known for both its creative and controversial potential.

        🎬 **DeepFake Studio** is an advanced platform designed for the detection and creation of deepfakes. Whether you're in the field of digital security 🔐, a digital artist 🎨 or a technology enthusiast 💡, DeepFake Studio equips you to explore and understand the depths and consequences of deepfakes.""")
    
    
    st.markdown("""<style>.margin {margin-top: 100px;}</style><div class="margin"></div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1: 
        st.markdown("""
         ### 🤖 How it works

        - 🔍 Select a mode from the sidebar to begin your exploration.
        - 🕵️‍♂️ In detector mode, load an image or video to analyze its veracity.
        - 🧑‍🎨 In generator mode, engage in the creation of deepfakes in compliance with legislation and ethical standards.""")
    with col2 : 
        st.markdown("""
        ### 💻 Technologies used

        At the cutting edge of innovation, we leverage machine learning and sophisticated neural networks 🧠 to guarantee optimum accuracy. """)
    

    # Use of empty space
    st.markdown("""<style>.margin {margin-top: 100px;}</style><div class="margin"></div>""", unsafe_allow_html=True)
    st.markdown("""
            ### 🛡️ Ethical and responsible use awareness

            We firmly believe in the ethical use of deepfakes. The power of this technology implies great responsibility. DeepFake Studio is committed to the fight against misinformation and manipulation. We call on our community to use these tools with integrity and discernment. 🤝 For each creation, ask yourself if it respects the privacy and dignity of individuals.""")
    
    st.markdown("""<style>.margin {margin-top: 50px;}</style><div class="margin"></div>""", unsafe_allow_html=True)
    st.markdown(""" 
            ### 🚀 Launch yourself ethically into the era of deepfakes with DeepFake Studio

            We invite you to dive into DeepFake Studio's various features. Discover the potential and limits of deepfakes while remaining vigilant about their impact on our society. 🌟""")
    

    st.markdown("## Examples")
    st.write("""
        Deepfakes can be used for a variety of purposes, from entertainment to education. Here are just a few examples where deepfakes have been applied creatively and responsibly:
        
        - **Film and Entertainment:** To recreate deceased actors or rejuvenate actors in flashback scenes.
        - **Education and Training:** Simulate historical conversations with virtually recreated iconic figures.
        - **Audio and Music:** Generate songs in the voices of famous singers, with permission.
        - **Customer Service:** To create virtual assistants who can provide a more human-like interaction.

        """)

    image_path_1 = "imageApp/exemple.jpg"
    st.image(image_path_1, caption='Swap realised by Deepfake Studio', use_column_width=True)
    st.markdown("""_**Note**: The above example is presented to inspire positive use of deepfakes. We always recommend respecting copyright and considering the ethical consequences of any deepfake project._""")
  

def get_prediction(image_path):
    # Charger l'image
    open_image = cv2.imread(image_path)
    open_image = cv2.cvtColor(open_image, cv2.COLOR_BGR2RGB)

    # Charger le détecteur de visages
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Convertir l'image en niveaux de gris pour la détection des visages
    gray_image = cv2.cvtColor(open_image, cv2.COLOR_RGB2GRAY)

    # Détecter les visages dans l'image
    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # S'il n'y a aucun visage détecté, retournez une erreur
    if len(faces) == 0:
        return "No face detected."

    # Prendre seulement le premier visage détecté
    (x, y, w, h) = faces[0]

    # Définir une marge autour de la zone du visage
    margin = 60
    x_margin = max(x - margin, 0)
    y_margin = max(y - margin, 0)
    w_margin = min(w + 2 * margin, open_image.shape[1])
    h_margin = min(h + 2 * margin, open_image.shape[0])

    # Extraire la région du visage avec la marge
    face_with_margin = open_image[
        y_margin : y_margin + h_margin, x_margin : x_margin + w_margin
    ]

    st.subheader("Face Detection")

    # Afficher l'image avec le rectangle autour du visage
    st.image(face_with_margin, caption="Face")

    # Redimensionner l'image à la taille attendue par le modèle
    resized_image = cv2.resize(face_with_margin, (256, 256))

    # Normaliser les valeurs de pixels
    resized_image = resized_image.astype("float") / 255.0

    # Ajouter une dimension pour correspondre à la forme attendue par le modèle
    input_image = np.expand_dims(resized_image, axis=0)

    # Faire la prédiction
    predicted_prob = predict_image(input_image)[0][0]

    # Déterminer la classe prédite et la confiance associée
    if predicted_prob >= 0.98:

        st.markdown(
            """
            <style>
            span.blue_bold {
                font-weight: bold;
                font-size:50px !important;
                color: blue;
            }
            </style>

            <span class="blue_bold">Real</span>
            """,
            unsafe_allow_html=True,
        )
        return ""

        # return f"Real, Confidence: {str(predicted_prob)[:4]}"
    else:

        st.markdown(
            """
            <style>
            span.blue_bold {
                font-weight: bold;
                font-size:50px !important;
                color: red;
            }
            </style>

            <span class="blue_bold">Fake</span>
            """,
            unsafe_allow_html=True,
        )

        return ""

        # return f"Fake, Confidence: {str(1 - predicted_prob)[:4]}"


def detector_mode():
    st.info("⚠️ Warning on the Use of Deepfakes: The use of deepfake technology can have important ethical and legal implications. Use this tool responsibly and do not create content that could mislead or harm others. Be sure to respect image rights and copyrights.")
    st.header("Deepfake Image Detector 🔍")
    st.markdown("Welcome to the Image Detector Mode, where the line between genuine and manipulated images is clarified. In this mode, our application employs sophisticated AI algorithms to accurately detect alterations and deepfakes in photographs.")

    st.write("### Instructions :")
    st.markdown("""
        Follow these simple steps to analyze your images:

        1. Click on 'Browse files' to upload an image from your device.
        2. Wait for the model to analyze the image and return the results.
    
        """)

    # upload an image
    uploaded_image = st.file_uploader(
        "Upload your own image to test the model:", type=["jpg", "jpeg", "png"]
    )

    # when an image is uploaded, display image and run inference
    if uploaded_image is not None:

        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_image.getvalue())

        # Affiche l'image téléchargée
        st.image("temp_image.jpg")

        if st.button("Predict"):

            progress_bar = st.progress(0)

            # Mettre à jour la barre de progression pendant que le modèle génère la prédiction
            for percent_complete in range(0, 101, 10):
                progress_bar.progress(percent_complete)
                time.sleep(0.5)

            st.text(get_prediction("temp_image.jpg"))

            os.remove("temp_image.jpg")


def plot_faces(img, faces):

    num_faces = len(faces)
    if num_faces == 0:
        st.write("No faces detected.")
        return

    if num_faces == 1:
        fig, axs = plt.subplots(figsize=(1, 1))
        bbox = faces[0]["bbox"]
        bbox = [int(b) for b in bbox]
        axs.imshow(img[bbox[1] : bbox[3], bbox[0] : bbox[2], ::-1])
        axs.axis("off")
    else:
        fig, axs = plt.subplots(1, num_faces, figsize=(12, 5))
        for i, face in enumerate(faces):
            bbox = face["bbox"]
            bbox = [int(b) for b in bbox]
            axs[i].imshow(img[bbox[1] : bbox[3], bbox[0] : bbox[2], ::-1])
            axs[i].text(
                0.5,
                1.05,
                "Face {}".format(i + 1),
                horizontalalignment="center",
                verticalalignment="bottom",
                transform=axs[i].transAxes,
            )
            axs[i].axis("off")

    st.pyplot(fig)


def save_img():

    with open("result.jpeg", "rb") as file:
        st.download_button(
            label="Download Result",
            data=file,
            file_name="result.jpeg",
            mime="image/jpeg",
        )


def swap_mode():

    st.info("⚠️ Warning on the Use of Deepfakes: The use of deepfake technology can have important ethical and legal implications. Use DeepScan responsibly and do not create content that could mislead or harm others. Be sure to respect image rights and copyrights.")
    st.header("Deepfake Image Generator 🖼️ ")
    st.markdown("""Welcome to the Swap Mode, where the lines between reality and digital artistry blur. In this mode, our application harnesses advanced AI algorithms to seamlessly exchange faces in images. """)

    st.write("### Instructions :")
    st.markdown("""
        Follow these simple steps to generate your deepfake:

        1. Click on 'Browse files' to upload an image from your device.
        2. Wait for the model to analyze the image and return the results.
""")

    # upload an image
    st.subheader("Upload Base Image")
    uploaded_image = st.file_uploader(
        "Image 1 :", type=["jpg", "jpeg", "png"]
    )

    res = None

    # when an image is uploaded, display image and run inference
    if uploaded_image is not None:

        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_image.getvalue())

        # Affiche l'image téléchargée
        st.image("temp_image.jpg")

        img = cv2.imread("temp_image.jpg")

        faces = detect_faces(img)

        if len(faces) <= 1:

            st.subheader("Upload Image to Swap")

            uploaded_image2 = st.file_uploader(
                "Image 2",
                type=["jpg", "jpeg", "png"],
            )

            if uploaded_image2 is not None:

                with open("temp_image2.jpg", "wb") as f:
                    f.write(uploaded_image2.getvalue())

                st.image("temp_image2.jpg")
                img2 = cv2.imread("temp_image2.jpg")

                faces2 = detect_faces(img2)

                if len(faces2) > 1:
                    st.subheader("All Faces Detection")
                    plot_faces(img2, faces2)
                    selected_face_index = st.selectbox(
                        "Select a face to swap:",
                        [None] + list(range(1, len(faces2) + 1)),
                    )

                    if selected_face_index is not None:
                        if st.button("Generate"):
                            res = swap_one(
                                img, faces[0], faces2[selected_face_index - 1]
                            )
                    else:
                        res = None

                elif len(faces2) == 1:
                    if st.button("Generate"):
                        res = swap_one(img, faces[0], faces2[0])
                else:
                    res = None

                if res is not None:

                    st.subheader("Result")

                    cv2.imwrite("result.jpeg", res)
                    st.image("result.jpeg")

                    save_img()
                    os.remove("result.jpeg")
                os.remove("temp_image2.jpg")

        else:

            st.subheader("All Faces Detection")

            plot_faces(img, faces)

            st.subheader("Select Face to Swap")

            selected_face_index = st.selectbox(
                "Select a face:", [None] + list(range(1, len(faces) + 1))
            )

            select_option = st.selectbox(
                "Swap mode:", [None, "All image", "Select Another Image"]
            )

            if select_option is not None:

                if select_option == "All image":

                    if selected_face_index is not None:
                        if st.button("Generate"):
                            res = swap_all(img, faces, selected_face_index - 1)
                            cv2.imwrite("result.jpeg", res)

                            st.subheader("Result")
                            st.image("result.jpeg")

                            save_img()
                            os.remove("result.jpeg")

                elif select_option == "Select Another Image":

                    st.subheader("Upload Second Image to Swap")

                    uploaded_image2 = st.file_uploader(
                        "Upload your own image 2 to test the model:",
                        type=["jpg", "jpeg", "png"],
                    )

                    if uploaded_image2 is not None:

                        with open("temp_image2.jpg", "wb") as f:
                            f.write(uploaded_image2.getvalue())

                        st.image("temp_image2.jpg")
                        img2 = cv2.imread("temp_image2.jpg")

                        faces2 = detect_faces(img2)

                        if len(faces2) > 1:
                            st.subheader("All Faces Detection")
                            plot_faces(img2, faces2)
                            selected_face_index2 = st.selectbox(
                                "Select a face to swap:",
                                [None] + list(range(1, len(faces2) + 1)),
                            )

                            if selected_face_index2 is not None:
                                if st.button("Generate"):
                                    res = swap_one(
                                        img,
                                        faces[selected_face_index - 1],
                                        faces2[selected_face_index2 - 1],
                                    )
                            else:
                                res = None

                        elif len(faces2) == 1:
                            if st.button("Generate"):
                                res = swap_one(
                                    img, faces[selected_face_index - 1], faces2[0]
                                )
                        else:
                            res = None

                        if res is not None:

                            st.subheader("Result")

                            cv2.imwrite("result.jpeg", res)
                            st.image("result.jpeg")

                            save_img()
                            os.remove("result.jpeg")
                        os.remove("temp_image2.jpg")

        os.remove("temp_image.jpg")


def voice_generator_mode():

    st.info("⚠️ Warning on the Use of Deepfakes: The use of deepfake technology can have important ethical and legal implications. Use this tool responsibly and do not create content that could mislead or harm others. Be sure to respect image rights and copyrights.")
    st.header("Deepfake Audio Generator 🎙️")
    st.markdown("Welcome to the Audio Generator Mode, where the distinction between real and synthetic voices fades away. In this mode, our application leverages cutting-edge AI algorithms to create realistic audio clips that mimic human speech.")

    st.write("### Instructions :")
    st.markdown("""
        Follow these simple steps to generate your deepfake:

        1. Click on 'Browse files' to upload an audio file from your device.
        2. Write the text you want to generate
        3. Wait for the model to analyze the text and the audio and return the results.
    
        """)
    st.subheader("Upload an audio file")

    # Charger un fichier audio
    """audio_file = st.file_uploader("Audio file", type=["mp3", "wav", "ogg"])

    # Vérifier si un fichier audio est chargé
    if audio_file is not None:
        with open("temp_audio." + audio_file.name.split(".")[-1], "wb") as f:
            f.write(audio_file.getvalue())

        # Afficher le fichier audio
        st.audio("temp_audio." + audio_file.name.split(".")[-1])

        st.subheader("Enter a text to generate")

        text = st.text_input("text", placeholder="Write sentence here !")

        if text != "":

            if st.button("Generate"):

                generate_audio(text, "temp_audio." + audio_file.name.split(".")[-1])

                st.subheader("Result")

                st.audio("output.wav")

                os.remove("temp_audio." + audio_file.name.split(".")[-1])

                with open("output.wav", "rb") as audio_file:
                    st.download_button(
                        "Download Generated Audio",
                        audio_file,
                        file_name="output.wav",
                        mime="audio/wav",
                    )

                os.remove("output.wav")"""


def video_generation():

    # chemins des fichiers en local
    config_file = "deepfake-video-generator/config.yml"  # remplacer par le chemin réel du fichier config
    model_weights = "deepfake-video-generator/model_weights.tar"  # remplacer par le chemin réel du fichier config

    # interface utilisateur pour uploader les fichiers
    st.info("⚠️ Warning on the Use of Deepfakes: The use of deepfake technology can have important ethical and legal implications. Use this tool responsibly and do not create content that could mislead or harm others. Be sure to respect image rights and copyrights.")
    st.header("Deepfake Video Generator 🎬")
    st.markdown("""Welcome to the Video Generator Mode, where the boundaries between reality and digital creation become fluid. In this mode, our application utilizes advanced AI algorithms to seamlessly generate or alter faces and scenes in videos.""")

    st.write("### Instructions :")
    st.markdown("""
        Follow these simple steps to generate your deepfake:

        1. Click on 'Browse files' to upload a source image and a driver video from your device.
        2. Click on 'Generate' button
        3. Wait for the model to analyze the files and return the results.
    
        """)
    st.subheader("Upload source image")
    source_image = st.file_uploader("Upload Source Image", type=["png", "jpg", "jpeg"])

    if source_image is not None:
        st.image(source_image)

        st.subheader("Upload driver video")
        driver_video = st.file_uploader("Upload Driver Video", type=["mp4"])

        if driver_video is not None:

            st.video(driver_video)

            # specification des nomms des fichiers de sortie
            output_video = "generated_video.mp4"
            output_video_fast = "generated_video_fast.mp4"

            # barre de progrewssion
            progress_bar = st.progress(0)

            def generate_animation(source_path, driver_path):
                # execution de la fonction de generation
                deep_fake_animation(
                    source_path,
                    driver_path,
                    config_file,
                    model_weights,
                    output_video,
                    output_video_fast,
                )
                # appelé une fois que la generation est terminée
                progress_bar.empty()  # nettoie la barre de progression

            # bouton pour lancer le processus de generation de la videao
            if st.button("Generate"):
                st.markdown("""_**Note** : The generation of the video may take some time (approximatively 15 minutes depending on the length of the video). Please be patient !_""")
                if source_image and driver_video:
                    # enregistrement des fichiers uploadés sur le serveur
                    with open("temp_source_image.png", "wb") as f:
                        f.write(source_image.getvalue())
                    # st.image("temp_source_image.png")
                    with open("temp_driver_video.mp4", "wb") as f:
                        f.write(driver_video.getvalue())
                    # st.video("temp_driver_video.mp4")

                    # demarage du processus de generation dans un thread sseparé
                    thread = threading.Thread(
                        target=generate_animation,
                        args=(
                            "temp_source_image.png",
                            "temp_driver_video.mp4",
                        ),
                    )
                    thread.start()

                    # tant que le thread est actif, maj de la barre de progression
                    while thread.is_alive():
                        time.sleep(0.1)  # maj toutes les 0.1 secondes
                        progress_bar.progress(
                            50
                        )  # maj avec la progression réelle si possible
                    thread.join()  # attendre que le trhead soit termine

                    # affichage de la vidéo generee si elle est disponible
                    if os.path.isfile(output_video_fast):
                        st.video(output_video_fast)
                        # utilisation de la fonction de sauvegarde pour télécharger la vidéo
                        with open(output_video_fast, "rb") as video_file:
                            st.download_button(
                                "Download Generated Video",
                                video_file,
                                file_name=output_video_fast,
                                mime="video/mp4",
                            )

                        os.remove("temp_source_image.png")
                        os.remove("temp_driver_video.mp4")
                        os.remove(output_video_fast)
                        # os.remove(video_file)

                else:
                    st.error(
                        "Please upload the source image and driver video to proceed."
                    )


st.sidebar.title('Deepfake Studio')
page = st.sidebar.selectbox(
    "Select Mode",
    ["", "Image Detector Mode", 
     "Image Generator Mode", 
     "Audio Generator Mode", 
     "Video Generator Mode"],
    index=0
)
if page == "":
    home()
elif page == "Image Detector Mode":
    detector_mode()
elif page == "Image Generator Mode":
    swap_mode()
elif page == "Audio Generator Mode":
    voice_generator_mode()
elif page == "Video Generator Mode":
    video_generation()
