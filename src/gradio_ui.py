import gradio as gr
from ask_capsyl_bot import chat_with_photos, chat_with_photos_2  # import your main bot function
from PIL import Image, ImageOps
import os

# Folder where photos are stored
PHOTO_ROOT = "/mnt/e/Google_Photos/InnoJam_Photos_Downsized"  # adjust if you have multiple folders
# PHOTO_ROOT = "/home/jovyan/capsyl_bot/data/images/InnoJam_Photos_Downsized"


def resize_long_side(img: Image.Image, target_size=500) -> Image.Image:
    width, height = img.size
    if width >= height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))
    return img.resize((new_width, new_height))


def run_bot(user_message, sim_threshold):
    bot_reply, results = chat_with_photos(user_message, sim_threshold, top_k=20)

    images_only = []
    for row in results:
        photo_name = row["photo_name"]
        # year = photo_name[:4]
        photo_path = os.path.join(PHOTO_ROOT, photo_name)

        if os.path.exists(photo_path):
            img = Image.open(photo_path).convert("RGB")
            img = ImageOps.exif_transpose(img)
            # img = auto_orient(img)  # fix EXIF rotation
            img = resize_long_side(img, 300)  # resize long side
            # img = fix_portrait(img)  # rotate if physically on side
            images_only.append(img)

    return bot_reply, images_only


# Build Gradio UI
with gr.Blocks(
    css="""
    .gradio-container {
        width: 60% !important;      /* 60% of the viewport width */
        max-width: 1200px;          /* optional max width */
        margin-left: auto;
        margin-right: auto;         /* center horizontally */
    }
"""
) as demo:
    gr.Markdown(
        """
        <div style="text-align:center">
          <h1>🖼️ Ask Capsyl Smart Chatbot</h1>
          <p>Your AI photo assistant for finding and exploring memories!</p>
        </div>
        """
    )

    query_input = gr.Textbox(
        label="Ask me about your photos", placeholder="e.g., Show me photos of Alice in Paris 2022"
    )
    threshold_slider = gr.Slider(
        minimum=0.0, maximum=1.0, value=0.2, step=0.01, label="Similarity Threshold"  # default
    )
    search_button = gr.Button("Search")
    output_text = gr.Textbox(
        label="Bot Response",
        lines=10,  # number of visible lines
        placeholder="Bot will reply here...",
        interactive=False,  # usually set False for output boxes
    )

    output_images = gr.Gallery(label="Photos", show_label=True, elem_id="gallery", columns=4, height="auto")

    search_button.click(
        fn=run_bot,
        inputs=[query_input, threshold_slider],
        outputs=[output_text, output_images],
    )

    # user_input.submit(run_bot, inputs=[query_input, threshold_slider], outputs=[output_text, output_images])


if __name__ == "__main__":
    demo.launch()
