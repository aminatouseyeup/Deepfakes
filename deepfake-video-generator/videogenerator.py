import os
import subprocess


def deep_fake_animation(
    source_image,
    driver_video,
    config_file,
    model_weights,
    output_video,
    output_video_fast,
):
    # Génération de l'animation DeepFake
    def deep_animate(source, driver, config, weights):
        command = ["deep_animate", str(source), str(driver), str(config), str(weights)]
        subprocess.run(command, check=True)

    # Obtention de la longueur d'une vidéo
    def get_length(filename):
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                filename,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return float(result.stdout)

    # Ajustement de la vitesse de la vidéo générée
    def adjust_video_speed(input_video, output_video, ratio):
        command = (
            f'ffmpeg -i {input_video} -filter:v "setpts={ratio}*PTS" {output_video}'
        )
        os.system(command)

    # Étape 1 : Génération de l'animation DeepFake
    deep_animate(source_image, driver_video, config_file, model_weights)

    # Étape 2 : Calcul du ratio des longueurs des vidéos
    driver_length = get_length(driver_video)
    deep_fake_length = get_length(output_video)
    ratio = round(driver_length / deep_fake_length, 4)

    # Étape 3 : Ajustement de la vitesse de la vidéo générée
    adjust_video_speed(output_video, output_video_fast, ratio)

    print(
        f"Traitement terminé : la vidéo ajustée est sauvegardée sous {output_video_fast}"
    )


# Exemple d'utilisation
# deep_fake_animation("source2.png", "driver.mp4", "config.yml", "model_weights.tar", "generated_video.mp4", "generated_video_fast.mp4")
