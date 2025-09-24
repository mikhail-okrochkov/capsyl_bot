import gradio as gr
from ask_capsyl_bot import chat_with_photos  # import your main bot function
from PIL import Image, ExifTags
import os

# Folder where photos are stored
PHOTO_ROOT = "/mnt/e/Google_Photos/"  # adjust if you have multiple folders


def auto_orient(img: Image.Image) -> Image.Image:
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = img._getexif()
        if exif is not None:
            ori_val = exif.get(orientation, None)
            if ori_val == 3:
                img = img.rotate(180, expand=True)
            elif ori_val == 6:
                img = img.rotate(270, expand=True)
            elif ori_val == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img


def fix_portrait(img: Image.Image) -> Image.Image:
    width, height = img.size
    # If width > height but intended as portrait, rotate
    if width > height:
        img = img.rotate(-90, expand=True)
    return img


def get_orientation(img: Image.Image) -> str:
    width, height = img.size
    if width > height:
        return "landscape"
    elif height > width:
        return "portrait"
    else:
        return "square"


def resize_long_side(img: Image.Image, target_size=500) -> Image.Image:
    width, height = img.size
    if width >= height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))
    return img.resize((new_width, new_height))


def run_bot(user_message):
    bot_reply, results = chat_with_photos(user_message, top_k=5)

    images_only = []
    for row in results:
        photo_name = row["photo_name"]
        year = photo_name[:4]
        photo_path = os.path.join(PHOTO_ROOT, f"Photos_from_{year}", photo_name)

        if os.path.exists(photo_path):
            img = Image.open(photo_path).convert("RGB")
            # img = auto_orient(img)  # fix EXIF rotation
            img = resize_long_side(img, 500)  # resize long side
            img = fix_portrait(img)  # rotate if physically on side
            images_only.append(img)

    return bot_reply, images_only


# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Capsyl Photo Chatbot")

    user_input = gr.Textbox(label="Ask me about your photos", placeholder="e.g., Show me photos of Alice in Paris 2022")
    output_text = gr.Textbox(
        label="Bot Response",
        lines=10,  # number of visible lines
        placeholder="Bot will reply here...",
        interactive=False,  # usually set False for output boxes
    )

    output_images = gr.Gallery(label="Photos", show_label=True, elem_id="gallery", columns=4, height="auto")

    user_input.submit(run_bot, inputs=user_input, outputs=[output_text, output_images])


if __name__ == "__main__":
    demo.launch()
