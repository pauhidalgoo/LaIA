import os
from moviepy.editor import ImageClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
from textwrap import fill

# Define directories and files
image_folder = "generated_images"
audio_file = "final_conversation.wav"
output_video = "final_video_with_subtitles.mp4"
image_duration = 10  # Duration of each image in seconds

# Time stamps for each line with their corresponding text
time_stamps = [
    (0.00, 5.61, "Bon dia, LaIA. M'agradaria saber com funciona el procés de sol·licitud de la beca Equitat."),
    (5.61, 15.89, "Bon dia, Cai. La beca Equitat és una ajuda econòmica que s'ofereix a estudiants de grau i màster habilitant de les universitats públiques de Catalunya, la UOC i alguns centres adscrits."),
    (15.89, 18.38, "Com funciona el procés de sol·licitud?"),
    # ... (inserta les altres línies aquí) ...
    (119.62, 121.46, "Moltes gràcies, LaIA."),
    (121.46, 125.81, "De res, Cai. Si tens més preguntes, no dubtis a consultar-me.")
]

# Load audio
audio = AudioFileClip(audio_file)
audio_duration = audio.duration

# Get list of image files and check if there are any images
image_files = sorted([
    os.path.join(image_folder, img)
    for img in os.listdir(image_folder)
    if img.endswith(".png") or img.endswith(".jpg")
])

if len(image_files) == 0:
    raise ValueError("No images found in the 'generated_images' folder.")

# Create video clips for each image and repeat them to cover the full audio duration
image_clips = []
for i in range(0, int(audio_duration // image_duration) + 1):
    img_path = image_files[i % len(image_files)]  # Repeat images in loop
    img_clip = ImageClip(img_path).set_duration(image_duration).resize(height=720)
    image_clips.append(img_clip)

# Concatenate all image clips into a background video that covers the full audio duration
background_video = concatenate_videoclips(image_clips, method="compose").set_duration(audio_duration)

# Create video clips with text for each segment in time_stamps
text_clips = []
max_text_width = 40  # Define max width for text wrapping

for start, end, text in time_stamps:
    # Format text with line breaks
    formatted_text = fill(text, width=max_text_width)  # Wrap text at a certain width

    # Create a TextClip for the dialogue line
    text_clip = TextClip(formatted_text, fontsize=24, color='white', font="Montserrat-Bold",
                         bg_color='rgba(0, 0, 0, 0.5)', size=(background_video.w, None), method="caption")
    text_clip = text_clip.set_duration(end - start).set_position(("center", "center")).set_start(start)

    text_clips.append(text_clip)

# Combine background video and text clips
final_video = CompositeVideoClip([background_video] + text_clips).set_audio(audio).set_fps(24)

# Export final video
final_video.write_videofile(output_video, codec="libx264", audio_codec="aac")

print(f"Video with subtitles created successfully and saved as '{output_video}'")