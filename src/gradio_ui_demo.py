import gradio as gr
from ask_capsyl_bot import chat_with_photos  # import your main bot function
from PIL import Image, ImageOps
import os
import pandas as pd
import numpy as np
import time

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


# --- Sequential "fake" responses ---
def run_bot_sequential(user_message, sim_threshold, counter):
    """
    Returns pre-defined bot replies and images in sequence.
    counter: gr.State to keep track of which response to show next.
    """
    # Load metadata once (can also move outside if large dataset)
    df = pd.read_excel("../data/google_photos_metadata_with_location.xlsx")

    # Define all “fake” responses as dictionaries
    # Each entry has the photo indices to show + the text to display
    responses = [
        {
            "indices": [647, 648, 649, 652, 653, 658, 659, 692, 693],
            "text": """I found these photos of Alesia in Hawaii:


Want me to group these into an album or narrow to a specific day (Feb 8, 9, or 13, 2025) or scene (beach, cliff, mountain)?""",
        },
        #         {
        #             "indices": [42,43,44,45],
        #             "text": """I found these photos of Alesia in Versailles, France on June 18th 2023:
        # """
        #         },
        {"indices": [110, 111, 112, 113], "text": """I found the following photos of Mikhail riding a bike indoors."""},
        {
            "indices": [
                64,
                65,
                130,
                397,
                398,
                669,
                671,
                672,
                675,
                676,
                710,
                711,
                714,
                715,
                716,
                683,
                684,
                685,
                763,
                773,
            ],
            "text": """I found 20 sunset photos in your library. Here they are, grouped by location and date:

- Hawaii
  - Feb 10, 2025: 20250210_182706.jpg, 20250210_182723.jpg, 20250210_182725.jpg, 20250210_183052.jpg
  - Feb 11, 2025: 20250211_071557.jpg
  - Feb 12, 2025: 20250212_180456.jpg, 20250212_180821.jpg, 20250212_181932.jpg
  - Feb 14, 2025 (Haleakalā): 20250214_181218.jpg, 20250214_181236.jpg, 20250214_182100.jpg, 20250214_182624.jpg, 20250214_182757.jpg

- San Diego
  - Apr 26–27, 2025: 20250426_191750.jpg, 20250427_192814.jpg

- Seattle
  - May 22, 2024: 20240522_205333.jpg, 20240522_205338.jpg

- Kusadasi, Turkiye
  - Jun 23, 2023: 20230623_202215.jpg, 20230623_202224.jpg

- Cancun, Mexico
  - Sep 5, 2023: 20230905_190705.jpg

Would you like me to open these, or create a “Sunsets” album with them?""",
        },
        {
            "indices": [663, 664, 665, 683, 684, 710, 711, 712],
            "text": """Here are your Hawaii sunset photos. I grouped them by location so you can pick what you want to view:

- Kauai (Poipu area):
  • 20250209_183126.jpg
  • 20250209_183128.jpg
  • 20250209_183139.jpg

- Maui – Kihei/Wailea beaches:
  • 20250212_180456.jpg
  • 20250212_180821.jpg

- Maui – Haleakalā summit:
  • 20250214_181218.jpg
  • 20250214_181236.jpg
  • 20250214_181515.jpg

Want me to open these, or create a “Hawaii Sunsets” album with them?""",
        },
    ]

    # Pick the response based on counter
    current = responses[counter % len(responses)]
    results = df.loc[current["indices"]]
    print(results)
    # Load images
    images_only = []
    for _, row in results.iterrows():
        photo_name = row["photo_name"]
        photo_path = os.path.join(PHOTO_ROOT, photo_name)

        if os.path.exists(photo_path):
            img = Image.open(photo_path).convert("RGB")
            img = ImageOps.exif_transpose(img)
            img = resize_long_side(img, 300)  # resize long side
            images_only.append(img)

    bot_reply = current["text"]

    sleep_time = 3 + np.random.rand() * 3
    time.sleep(sleep_time)  # simulate processing delay

    # increment counter for next click
    counter += 1

    return bot_reply, images_only, counter


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
        minimum=0.0, maximum=1.0, value=0.2, step=0.01, label="Similarity Threshold", visible=False  # default
    )
    # search_button = gr.Button("Search")
    output_text = gr.Textbox(
        label="Bot Response",
        lines=10,  # number of visible lines
        placeholder="Bot will reply here...",
        interactive=False,  # usually set False for output boxes
    )

    output_images = gr.Gallery(label="Photos", show_label=True, elem_id="gallery", columns=4, height="auto")

    state_counter = gr.State(value=0)

    query_input.submit(
        fn=run_bot_sequential,
        inputs=[query_input, threshold_slider, state_counter],
        outputs=[output_text, output_images, state_counter],
    )

    # user_input.submit(run_bot, inputs=[query_input, threshold_slider], outputs=[output_text, output_images])


if __name__ == "__main__":
    demo.launch()
