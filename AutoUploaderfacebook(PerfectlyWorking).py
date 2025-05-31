# --- START OF COMBINED SCRIPT: video_creator_uploader.py ---

import os
import subprocess
import shlex
import uharfbuzz as hb
import freetype
from PIL import Image, ImageDraw
import re
from io import BytesIO
import random
import time
import requests # For Facebook upload

# --- SDKs for Gemini ---
import google.generativeai as gemini_text_sdk
try:
    from google import genai as gemini_image_sdk_base # Not strictly used for image gen in this script
    from google.genai import types as gemini_image_sdk_types # Not strictly used
    if not hasattr(gemini_image_sdk_base, 'Client'):
        print("Warning: 'gemini_image_sdk_base.Client' not found (though not used for image gen now).")
except ImportError as e:
    print(f"Warning: Could not import 'google.genai' or 'google.genai.types' (though not used for image gen now): {e}")
    gemini_image_sdk_base = None
    gemini_image_sdk_types = None

# === GLOBAL CONFIGURATION ===

# --- Gemini API Configuration ---
GEMINI_API_KEY = "AIzaSyDEKsW0GhR5pBHBcDVDz3xZSfZD1lkAvVM" # Your Gemini API Key

# --- Facebook API Configuration ---
FB_ACCESS_TOKEN = 'EAAPPMNJb8NUBO8YTimpZCZCz6tZAB8YE9XIZCaVchgZB3P7S80lLXZBO36fMJ0LLLrqJ1hzwOaH5QlNFurYZAbnTTZClPUNgW0avZB1vy72S5HSDWhHmYcrTaQQt3mGfYGNIdPEOPUZBduQ1Xkmg5HdnJDH8bmnyT2AfG9PUowBspSCxxJIBQ7JJoaZBEWeGXUQAeeZC'
FB_PAGE_ID = '603045946225330'

# --- Video Generation Configuration ---
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_DURATION_SECONDS = 10
VIDEO_FPS = 25

# Text Rendering Configuration
FONT_FILE_PATH_SANSKRIT = "NotoSerifDevanagari_ExtraCondensed-Bold.ttf"
FONT_FILE_PATH_HINDI = "NotoSerifDevanagari_ExtraCondensed-Bold.ttf"
FONT_FILE_PATH_ENGLISH = "NotoSerifDevanagari_ExtraCondensed-Bold.ttf" # Update if needed for better Latin

TEXT_COLOR_RGBA = (255, 255, 255, 255)
TARGET_TEXT_WIDTH_PERCENT = 0.88
LINE_SPACING_FACTOR = 1.2
SPACE_BETWEEN_BLOCKS_PT = 15

# Dynamic Font Size Configuration
BASE_FONT_SIZE_SANSKRIT = 60
MIN_FONT_SIZE_SANSKRIT = 38
IDEAL_LENGTH_FOR_BASE_SANSKRIT = 60
MAX_LENGTH_FOR_MIN_SANSKRIT = 150

BASE_FONT_SIZE_HINDI = 52
MIN_FONT_SIZE_HINDI = 32
IDEAL_LENGTH_FOR_BASE_HINDI = 90
MAX_LENGTH_FOR_MIN_HINDI = 220

BASE_FONT_SIZE_ENGLISH = 48
MIN_FONT_SIZE_ENGLISH = 28
IDEAL_LENGTH_FOR_BASE_ENGLISH = 120
MAX_LENGTH_FOR_MIN_ENGLISH = 280

# Pre-existing Background Image Configuration
PREEXISTING_BACKGROUND_IMAGE_FOLDER = "images" # <<< IMPORTANT: SET THIS PATH
ALLOWED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
AI_BACKGROUND_IMAGE_FILTER_OPACITY = 0.5
AI_GENERATED_BACKGROUND_FILENAME = "filtered_background_for_video.png"
IMAGE_INDEX_TRACKER_FILE = "last_image_index.txt"

# Shloka Uniqueness
USED_SHLOKAS_FILE = "used_shlokas.txt"
MAX_SHLOKA_RETRIES = 5

# Output files from creation process
RENDERED_TEXT_PNG = "rendered_gemini_text_overlay.png"
FINAL_OUTPUT_VIDEO = "gemini_video_with_ai_bg_v3.mp4" # This will be uploaded

# Background Music Configuration
BACKGROUND_MUSIC_FILES_LIST = [
    "audio1cor.mp3", "audio2cor.mp3", "audio3cor.mp3",
    "audio4cor.mp3", "audio5cor.mp3", "audio6cor.mp3", "audio7cor.mp3"
]

# Retry Configuration for Video Generation
MAX_OVERALL_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 60

# --- Initialize Gemini SDK ---
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found. Please set it in the script.")
    exit()
gemini_text_sdk.configure(api_key=GEMINI_API_KEY)


# === VIDEO CREATION MODULE HELPER FUNCTIONS (from ultimatetestforidexedimage.py) ===

def _apply_black_filter_to_pil_image(pil_image: Image.Image, opacity: float) -> Image.Image:
    if not (0 <= opacity <= 1):
        opacity = max(0, min(1, opacity))
    if opacity == 0:
        return pil_image.convert('RGB') if pil_image.mode != 'RGB' else pil_image
    base_image_rgba = pil_image.convert('RGBA') if pil_image.mode != 'RGBA' else pil_image.copy()
    alpha_value = int(255 * opacity)
    black_layer = Image.new('RGBA', base_image_rgba.size, (0, 0, 0, alpha_value))
    filtered_image = Image.alpha_composite(base_image_rgba, black_layer)
    return filtered_image.convert('RGB')

def get_gemini_text_response(prompt: str) -> str:
    print(f"Fetching text response from Gemini for prompt: '{prompt}'")
    try:
        model = gemini_text_sdk.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        if response.parts:
            return "".join(part.text for part in response.parts if hasattr(part, 'text')).strip()
        elif hasattr(response, 'text') and response.text:
             return response.text.strip()
        else:
            print("Gemini text response did not contain text.")
            return "Error: No text in Gemini response."
    except Exception as e:
        print(f"Error fetching text from Gemini: {e}")
        return f"Error: Could not fetch text. {e}"

def normalize_shloka_for_comparison(shloka_text: str) -> str:
    if not shloka_text: return ""
    text = shloka_text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    text = text.replace("\n", " ")
    return " ".join(text.split()).strip().lower()

def load_used_shlokas(filepath: str) -> set:
    used = set()
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    used.add(line.strip())
        except IOError as e:
            print(f"Warning: Could not read used shlokas file '{filepath}': {e}")
    return used

def add_shloka_to_used(shloka_key: str, used_shlokas_set: set, filepath: str):
    if shloka_key not in used_shlokas_set:
        used_shlokas_set.add(shloka_key)
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(shloka_key + "\n")
            print(f"Added shloka key '{shloka_key[:50]}...' to used list.")
        except IOError as e:
            print(f"Warning: Could not write to used shlokas file '{filepath}': {e}")

def split_shloka_hindi_english(full_text: str) -> tuple[str, str, str]:
    sanskrit, hindi, english = "", "", ""
    text = full_text.strip()
    eng_patterns = [
        r"(.*?)English Meaning:(.*)", r"(.*?)Eng Meaning:(.*)",
        r"(.*?)English Translation:(.*)", r"(.*?)English:(.*)"
    ]
    text_before_eng = text
    for p in eng_patterns:
        m = re.search(p, text, re.DOTALL | re.IGNORECASE)
        if m:
            text_before_eng = m.group(1).strip()
            english = re.sub(r"^(English Meaning:|Eng Meaning:|English Translation:|English:)\s*", "", m.group(2).strip(), flags=re.IGNORECASE).strip()
            break
    hindi_patterns = [
        r"(.*?)हिंदी अर्थ:(.*)", r"(.*?)Hindi Meaning:(.*)",
        r"(.*?)अर्थ:(.*)", r"(.*?)Meaning:(.*)"
    ]
    text_before_hindi = text_before_eng
    for p in hindi_patterns:
        m = re.search(p, text_before_eng, re.DOTALL | re.IGNORECASE)
        if m:
            text_before_hindi = m.group(1).strip()
            hindi = re.sub(r"^(हिंदी अर्थ:|Hindi Meaning:|अर्थ:|Meaning:)\s*", "", m.group(2).strip(), flags=re.IGNORECASE).strip()
            break
    sanskrit = text_before_hindi.strip()
    if sanskrit:
        sanskrit = re.sub(r"^\s*[\*#-_]+\s*(.*?)\s*[\*#-_]+\s*$", r"\1", sanskrit, flags=re.UNICODE).strip()
        sanskrit = re.sub(r"^\s*[\*]{2,}\s*(.*?)\s*[\*]{2,}\s*$", r"\1", sanskrit, flags=re.UNICODE).strip()
        sanskrit_prefix_patterns = [
            r"^(श्लोकः|Shloka:|Shlokah:|Shlok:|Sloka:|Verse:)\s*[:\*]*\s*",
        ]
        for prefix_pattern in sanskrit_prefix_patterns:
            sanskrit = re.sub(prefix_pattern, "", sanskrit, flags=re.IGNORECASE | re.UNICODE).strip()
        sanskrit = re.sub(r"^\s*[^a-zA-Z0-9\u0900-\u097F]+", "", sanskrit, flags=re.UNICODE).strip()
    temp_texts = {'sanskrit': sanskrit, 'hindi': hindi, 'english': english}
    for text_key in ['sanskrit', 'hindi', 'english']:
        current_text = temp_texts[text_key]
        if current_text:
            current_text = re.sub(r"^\s*[\*#-_]+\s*|\s*[\*#-_]+\s*$", "", current_text, flags=re.UNICODE).strip()
            current_text = re.sub(r"^\s*[\*]{2,}\s*(.*?)\s*[\*]{2,}\s*$", r"\1", current_text, flags=re.UNICODE).strip()
            temp_texts[text_key] = current_text
    sanskrit, hindi, english = temp_texts['sanskrit'], temp_texts['hindi'], temp_texts['english']
    if not sanskrit.strip() and (hindi.strip() or english.strip()):
        print("Warning: Parsed Sanskrit shloka is empty after all cleaning.")
    print(f"--- Cleaned Sanskrit Part ---\n{sanskrit}\n--------------------")
    print(f"--- Cleaned Hindi Meaning Part ---\n{hindi}\n-------------------")
    print(f"--- Cleaned English Meaning Part ---\n{english}\n-----------------")
    return sanskrit, hindi, english

def calculate_dynamic_font_size(text_length: int, base_font_size: int, min_font_size: int,
                               ideal_length_for_base: int, max_length_for_min: int) -> int:
    if text_length <= 0: return base_font_size
    if text_length <= ideal_length_for_base: return base_font_size
    if text_length >= max_length_for_min: return min_font_size
    if max_length_for_min <= ideal_length_for_base: return min_font_size # Should not happen with good config
    progress = (text_length - ideal_length_for_base) / (max_length_for_min - ideal_length_for_base)
    font_size_reduction = (base_font_size - min_font_size) * progress
    new_size = base_font_size - font_size_reduction
    return max(min_font_size, int(new_size))

def shape_text_line_hb(hb_font, text_segment):
    buf = hb.Buffer()
    buf.add_str(text_segment)
    buf.guess_segment_properties()
    features = {"kern": True, "liga": True, "clig": True, "calt": True}
    try: deva_script_tag = hb.ot_tag_to_script('deva')
    except TypeError: deva_script_tag = hb.ot_tag_to_script(b'deva') # Compatibility with older uharfbuzz
    if buf.script == deva_script_tag:
        features.update({
            "akhn": True, "rkrf": True, "blwf": True, "half": True, "pstf": True,
            "vatu": True, "pres": True, "abvs": True, "blws": True, "psts": True,
            "haln": True, "locl": True, "nukt": True, "pref": True, "cjct": True
        })
    hb.shape(hb_font, buf, features)
    return buf.glyph_infos, buf.glyph_positions

def get_line_width_px(glyph_positions):
    width = sum(pos.x_advance for pos in glyph_positions)
    return width // 64

def wrap_and_shape_text_block(text_block_content: str, ft_face_obj, hb_font_obj, font_size_pt_val, max_line_width_px_val, current_line_spacing_factor: float):
    if not text_block_content.strip(): return [], 0, 0
    ft_face_obj.set_char_size(font_size_pt_val * 64)
    hb_font_obj.scale = (font_size_pt_val * 64, font_size_pt_val * 64)
    explicit_lines = text_block_content.split('\n')
    all_shaped_lines_data = []
    max_overall_block_width_px = 0
    for line_segment in explicit_lines:
        words = line_segment.split()
        current_line_text_val = ""
        current_line_infos_shaped, current_line_positions_shaped = None, None
        for word in words:
            if not word: continue
            test_word_segment = (" " if current_line_text_val else "") + word
            potential_new_line_text = current_line_text_val + test_word_segment
            potential_line_infos_s, potential_line_positions_s = shape_text_line_hb(hb_font_obj, potential_new_line_text)
            potential_line_width_px = get_line_width_px(potential_line_positions_s)
            if potential_line_width_px <= max_line_width_px_val or not current_line_text_val:
                current_line_text_val = potential_new_line_text
                current_line_infos_shaped, current_line_positions_shaped = potential_line_infos_s, potential_line_positions_s
            else:
                if current_line_infos_shaped:
                    line_data = (current_line_infos_shaped, current_line_positions_shaped, get_line_width_px(current_line_positions_shaped))
                    all_shaped_lines_data.append(line_data)
                    max_overall_block_width_px = max(max_overall_block_width_px, line_data[2])
                current_line_text_val = word
                current_line_infos_shaped, current_line_positions_shaped = shape_text_line_hb(hb_font_obj, current_line_text_val)
        if current_line_text_val and current_line_infos_shaped:
            line_data = (current_line_infos_shaped, current_line_positions_shaped, get_line_width_px(current_line_positions_shaped))
            all_shaped_lines_data.append(line_data)
            max_overall_block_width_px = max(max_overall_block_width_px, line_data[2])
    ft_line_height_px = int(ft_face_obj.size.height // 64 * current_line_spacing_factor)
    total_block_height_px = len(all_shaped_lines_data) * ft_line_height_px
    if all_shaped_lines_data and len(all_shaped_lines_data) > 1: # Adjust for inter-line spacing within a block
        total_block_height_px -= int(ft_face_obj.size.height // 64 * (current_line_spacing_factor - 1.0))
    if not all_shaped_lines_data: total_block_height_px = 0
    return all_shaped_lines_data, max_overall_block_width_px, total_block_height_px

def render_text_to_png_manual_v3(sanskrit_text: str, hindi_text: str, english_text: str,
                                 output_png_path: str,
                                 eff_font_size_sanskrit: int, eff_font_size_hindi: int, eff_font_size_english: int):
    print(f"Rendering Skt({eff_font_size_sanskrit}pt), Hin({eff_font_size_hindi}pt), Eng({eff_font_size_english}pt) to {output_png_path}...")
    font_paths = {
        "sanskrit": FONT_FILE_PATH_SANSKRIT, "hindi": FONT_FILE_PATH_HINDI, "english": FONT_FILE_PATH_ENGLISH
    }
    for lang, path in font_paths.items():
        if not os.path.exists(path):
            print(f"ERROR: Font file for {lang} not found at '{path}'.")
            if lang == "english" and path == FONT_FILE_PATH_SANSKRIT: # Common misconfiguration hint
                 print(f"Hint: For English, FONT_FILE_PATH_ENGLISH is set to '{path}'. Update if needed.")
            return False
    try:
        ft_faces = {lang: freetype.Face(path) for lang, path in font_paths.items()}
        hb_fonts = {}
        for lang, path in font_paths.items():
            with open(path, 'rb') as f_data: font_data = f_data.read()
            hb_fonts[lang] = hb.Font(hb.Face(font_data))
        max_render_line_width_px = int(VIDEO_WIDTH * TARGET_TEXT_WIDTH_PERCENT)
        sanskrit_lines_data, sk_max_w, sk_h = wrap_and_shape_text_block(
            sanskrit_text, ft_faces["sanskrit"], hb_fonts["sanskrit"], eff_font_size_sanskrit, max_render_line_width_px, LINE_SPACING_FACTOR)
        hindi_lines_data, hi_max_w, hi_h = wrap_and_shape_text_block(
            hindi_text, ft_faces["hindi"], hb_fonts["hindi"], eff_font_size_hindi, max_render_line_width_px, LINE_SPACING_FACTOR)
        english_lines_data, en_max_w, en_h = wrap_and_shape_text_block(
            english_text, ft_faces["english"], hb_fonts["english"], eff_font_size_english, max_render_line_width_px, LINE_SPACING_FACTOR)
        total_canvas_width = max(sk_max_w, hi_max_w, en_max_w, 1) + 40 # Add padding
        space_px = int((SPACE_BETWEEN_BLOCKS_PT / 72) * 96 * (max(eff_font_size_sanskrit, eff_font_size_hindi, eff_font_size_english, 12) / 12.0))
        num_blocks_with_content = 0
        if sk_h > 0 : num_blocks_with_content +=1
        if hi_h > 0 : num_blocks_with_content +=1
        if en_h > 0 : num_blocks_with_content +=1
        total_spacing_px = space_px * (num_blocks_with_content - 1) if num_blocks_with_content > 1 else 0
        total_canvas_height = sk_h + hi_h + en_h + total_spacing_px + 40 # Add padding
        if total_canvas_height <= 40: print("No text content with height to render."); return False
        image = Image.new('RGBA', (int(total_canvas_width), int(total_canvas_height)), (0, 0, 0, 0)) # Transparent background
        current_y_offset = 20 # Initial top padding
        text_blocks_to_render = [
            (sanskrit_text, sanskrit_lines_data, ft_faces["sanskrit"], eff_font_size_sanskrit, sk_h),
            (hindi_text, hindi_lines_data, ft_faces["hindi"], eff_font_size_hindi, hi_h),
            (english_text, english_lines_data, ft_faces["english"], eff_font_size_english, en_h),
        ]
        first_block_rendered = False
        for text_content, lines_data, ft_face, font_size, block_height in text_blocks_to_render:
            if block_height > 0:
                if first_block_rendered: current_y_offset += space_px
                ft_face.set_char_size(font_size * 64)
                line_height_px = int(ft_face.size.height // 64 * LINE_SPACING_FACTOR)
                ascender_px = (ft_face.ascender / ft_face.units_per_EM) * font_size if ft_face.units_per_EM else (ft_face.size.ascender // 64)
                y_baseline = current_y_offset + ascender_px
                for line_infos, line_positions, line_w_px in lines_data:
                    x_pen = (total_canvas_width - line_w_px) // 2 # Center line
                    for info, pos in zip(line_infos, line_positions):
                        ft_face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
                        bitmap = ft_face.glyph.bitmap
                        x_render = x_pen + (pos.x_offset // 64) + ft_face.glyph.bitmap_left
                        y_render = y_baseline - ft_face.glyph.bitmap_top + (pos.y_offset // 64)
                        if bitmap.width > 0 and bitmap.rows > 0:
                            data = [(TEXT_COLOR_RGBA[0], TEXT_COLOR_RGBA[1], TEXT_COLOR_RGBA[2], val) for val in bitmap.buffer]
                            glyph_img = Image.new("RGBA", (bitmap.width, bitmap.rows)); glyph_img.putdata(data)
                            image.paste(glyph_img, (int(x_render), int(y_render)), mask=glyph_img)
                        x_pen += pos.x_advance // 64
                    y_baseline += line_height_px
                current_y_offset += block_height
                first_block_rendered = True
        image.save(output_png_path)
        print(f"Successfully rendered text to {output_png_path}")
        return True
    except Exception as e_render:
        print(f"Error during manual text rendering: {e_render}")
        import traceback; traceback.print_exc()
        return False

def create_video_with_ai_bg_overlay_and_music(bg_image_path: str, png_overlay_path: str, music_path: str, out_vid_path: str):
    if not os.path.exists(bg_image_path): print(f"ERROR: Background image not found: {bg_image_path}"); return False
    if not os.path.exists(png_overlay_path): print(f"ERROR: PNG overlay not found: {png_overlay_path}"); return False
    inputs = [
        '-loop', '1', '-framerate', str(VIDEO_FPS), '-i', bg_image_path.replace('\\', '/'),
        '-i', png_overlay_path.replace('\\', '/')
    ]
    background_filter = (
        f"[0:v]scale='iw*max({VIDEO_WIDTH}/iw,{VIDEO_HEIGHT}/ih)':'ih*max({VIDEO_WIDTH}/iw,{VIDEO_HEIGHT}/ih)',"
        f"crop={VIDEO_WIDTH}:{VIDEO_HEIGHT},setsar=1[bg_scaled]"
    )
    filter_complex_parts = [
        background_filter,
        f"[bg_scaled][1:v]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2[outv]"
    ]
    filter_complex_str = ";".join(filter_complex_parts)
    maps = ['-map', '[outv]']
    output_opts = ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-r', str(VIDEO_FPS), '-t', str(VIDEO_DURATION_SECONDS)]
    final_cmd_opts = []
    music_exists_and_valid = music_path and os.path.exists(str(music_path).replace('\\', '/'))
    if music_exists_and_valid:
        inputs.extend(['-i', str(music_path).replace('\\', '/')])
        maps.extend(['-map', '2:a?'])
        output_opts.extend(['-c:a', 'aac', '-b:a', '192k'])
        final_cmd_opts.append('-shortest') # Ensure video duration is capped by shortest input (music)
    else:
        if music_path: print(f"Warning: Music file '{music_path}' not found.")
        else: print("No music file specified.")
    command = ['ffmpeg'] + inputs + ['-filter_complex', filter_complex_str] + maps + output_opts + final_cmd_opts + ['-y', out_vid_path]
    print(f"Executing FFmpeg command: {' '.join(shlex.quote(str(c)) for c in command)}")
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"Video created: {out_vid_path}")
        try:
            ffprobe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', out_vid_path]
            duration_output = subprocess.check_output(ffprobe_cmd, text=True)
            actual_duration = float(duration_output.strip())
            print(f"Verified video duration: {actual_duration:.2f} seconds.")
            target_dur = float(VIDEO_DURATION_SECONDS)
            if abs(actual_duration - target_dur) > 0.5 and not (music_exists_and_valid and actual_duration < target_dur):
                 print(f"WARNING: Video duration ({actual_duration:.2f}s) differs from target ({target_dur}s). Check music length if used.")
        except Exception as e_probe: print(f"Could not verify video duration: {e_probe}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}\nFFmpeg stdout:\n{e.stdout}\nFFmpeg stderr:\n{e.stderr}")
        return False
    except FileNotFoundError: print("Error: ffmpeg or ffprobe command not found. Ensure they are in your system's PATH."); return False

def get_next_image_index_and_update(tracker_file: str, num_available_images: int) -> int:
    if num_available_images == 0:
        return -1
    current_index = 0
    if os.path.exists(tracker_file):
        try:
            with open(tracker_file, 'r') as f:
                content = f.read().strip()
                if content.isdigit():
                    current_index = int(content)
                else:
                    print(f"Warning: Tracker file '{tracker_file}' contained non-integer content '{content}'. Resetting index to 0.")
        except IOError as e:
            print(f"Warning: Could not read tracker file '{tracker_file}': {e}. Resetting index to 0.")
        except ValueError:
            print(f"Warning: Tracker file '{tracker_file}' contained non-integer. Resetting index to 0.")

    index_to_use = current_index % num_available_images # Use current value, then update for next run
    next_index_to_store = (index_to_use + 1) % num_available_images

    try:
        with open(tracker_file, 'w') as f:
            f.write(str(next_index_to_store))
    except IOError as e:
        print(f"Warning: Could not write to tracker file '{tracker_file}': {e}")
    return index_to_use

# === MAIN VIDEO GENERATION PIPELINE (Modified to return video_path, title, description) ===
def run_video_generation_pipeline():
    """
    Runs the full video generation pipeline.
    Returns:
        tuple: (video_path, title, description) on success, (None, None, None) on failure.
    """
    print("--- Step 1: Selecting and Filtering Pre-existing Background Image (Sequentially) ---")
    if not os.path.isdir(PREEXISTING_BACKGROUND_IMAGE_FOLDER):
        print(f"Error: Background image folder not found: {PREEXISTING_BACKGROUND_IMAGE_FOLDER}")
        return None, None, None
    
    all_files_in_folder = os.listdir(PREEXISTING_BACKGROUND_IMAGE_FOLDER)
    image_files_unsorted = [
        f for f in all_files_in_folder
        if os.path.isfile(os.path.join(PREEXISTING_BACKGROUND_IMAGE_FOLDER, f)) and \
           os.path.splitext(f)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    ]
    if not image_files_unsorted:
        print(f"Error: No suitable image files in {PREEXISTING_BACKGROUND_IMAGE_FOLDER}")
        return None, None, None
    
    image_files_sorted = sorted(image_files_unsorted)
    num_images = len(image_files_sorted)
    current_image_idx = get_next_image_index_and_update(IMAGE_INDEX_TRACKER_FILE, num_images)
    
    if current_image_idx == -1 :
        print("Error: Could not determine image index (no images available).")
        return None, None, None
        
    selected_image_name = image_files_sorted[current_image_idx]
    selected_image_path = os.path.join(PREEXISTING_BACKGROUND_IMAGE_FOLDER, selected_image_name)
    print(f"Selected background image (Index {current_image_idx}): {selected_image_path}")

    try:
        pil_image_raw = Image.open(selected_image_path)
        print(f"Applying black filter with opacity {AI_BACKGROUND_IMAGE_FILTER_OPACITY}...")
        pil_image_filtered = _apply_black_filter_to_pil_image(pil_image_raw, AI_BACKGROUND_IMAGE_FILTER_OPACITY)
        
        original_width, original_height = pil_image_filtered.size
        new_width = int(original_width * 1.01) # Slight width increase
        print(f"Scaling filtered image width from {original_width} to {new_width} (1% increase)...")
        resampling_method = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        pil_image_scaled_width = pil_image_filtered.resize((new_width, original_height), resampling_method)
        
        os.makedirs(os.path.dirname(AI_GENERATED_BACKGROUND_FILENAME) or '.', exist_ok=True)
        pil_image_scaled_width.save(AI_GENERATED_BACKGROUND_FILENAME)
        print(f"Scaled and filtered background image saved to '{AI_GENERATED_BACKGROUND_FILENAME}'")
    except Exception as e:
        print(f"Error processing background image '{selected_image_path}': {e}")
        import traceback; traceback.print_exc()
        return None, None, None
    print("Background image processing successful.\n")

    print("--- Step 2: Fetching Unique Shloka and Meanings (Hindi, English) ---")
    used_shlokas_set = load_used_shlokas(USED_SHLOKAS_FILE)
    sanskrit_shloka_raw, hindi_meaning_raw, english_meaning_raw = "", "", ""
    text_fetch_success = False
    user_prompt_for_text = (
        "Provide a concise Sanskrit shloka (1-2 lines) from Hindu scriptures like the Upanishads, Bhagavad Gita, Mahabharata, Puranas, or other Smriti or Shruti texts."
        " Then, after 'हिंदी अर्थ:', provide an ACCURATE Hindi meaning (1-2 lines) for the shloka."
        " Then, after 'English Meaning:', provide its English meaning (1-2 lines)."
        " Keep all three parts relatively short and suitable for a vertical video format. Ensure the Hindi translation is a correct interpretation of the Sanskrit."
    )
    for shloka_attempt in range(MAX_SHLOKA_RETRIES):
        print(f"Shloka fetch attempt {shloka_attempt + 1}/{MAX_SHLOKA_RETRIES}...")
        gemini_text_response = get_gemini_text_response(user_prompt_for_text)
        if "Error:" in gemini_text_response or not gemini_text_response.strip():
            print(f"Gemini text error: {gemini_text_response}"); continue
        
        temp_skt, temp_hin, temp_eng = split_shloka_hindi_english(gemini_text_response)
        if not temp_skt.strip(): print("Parsed Sanskrit shloka empty. Retrying."); continue
        
        shloka_key_for_used_check = normalize_shloka_for_comparison(temp_skt.replace(",", " "))
        if shloka_key_for_used_check not in used_shlokas_set:
            sanskrit_shloka_raw, hindi_meaning_raw, english_meaning_raw = temp_skt, temp_hin, temp_eng
            add_shloka_to_used(shloka_key_for_used_check, used_shlokas_set, USED_SHLOKAS_FILE)
            text_fetch_success = True; print("Unique shloka and meanings obtained."); break
        else: print(f"Shloka key '{shloka_key_for_used_check[:50]}...' used. Requesting new one.")
            
    if not text_fetch_success: print("Failed to obtain unique shloka."); return None, None, None

    # Prepare texts for rendering (may differ slightly from raw for FB description)
    sanskrit_shloka_render = sanskrit_shloka_raw
    hindi_meaning_render = hindi_meaning_raw
    english_meaning_render = english_meaning_raw # Keep raw for FB, format for rendering if needed

    if sanskrit_shloka_render: sanskrit_shloka_render = sanskrit_shloka_render.replace(",", "\n")
    if hindi_meaning_render: hindi_meaning_render = hindi_meaning_render.replace(",", "\n")
    
    # Add quotes for rendering if they are actual spoken parts
    if sanskrit_shloka_render.strip(): sanskrit_shloka_render = f'"{sanskrit_shloka_render.strip()}"'
    if hindi_meaning_render.strip(): hindi_meaning_render = f'"{hindi_meaning_render.strip()}"'
    # English meaning usually is descriptive, quotes might not always be appropriate for rendering
    # if english_meaning_render.strip(): english_meaning_render = f'"{english_meaning_render.strip()}"'


    print(f"--- Final Processed Sanskrit (for rendering) ---\n{sanskrit_shloka_render}\n--------------------")
    print(f"--- Final Processed Hindi (for rendering) ---\n{hindi_meaning_render}\n-------------------")
    print(f"--- Final Processed English (for rendering) ---\n{english_meaning_render}\n-----------------")

    eff_font_size_sk = calculate_dynamic_font_size(len(sanskrit_shloka_render), BASE_FONT_SIZE_SANSKRIT, MIN_FONT_SIZE_SANSKRIT, IDEAL_LENGTH_FOR_BASE_SANSKRIT, MAX_LENGTH_FOR_MIN_SANSKRIT)
    eff_font_size_hi = calculate_dynamic_font_size(len(hindi_meaning_render), BASE_FONT_SIZE_HINDI, MIN_FONT_SIZE_HINDI, IDEAL_LENGTH_FOR_BASE_HINDI, MAX_LENGTH_FOR_MIN_HINDI)
    eff_font_size_en = calculate_dynamic_font_size(len(english_meaning_render), BASE_FONT_SIZE_ENGLISH, MIN_FONT_SIZE_ENGLISH, IDEAL_LENGTH_FOR_BASE_ENGLISH, MAX_LENGTH_FOR_MIN_ENGLISH)
    print(f"Calculated font sizes - Skt: {eff_font_size_sk}pt, Hin: {eff_font_size_hi}pt, Eng: {eff_font_size_en}pt\n")

    print(f"--- Step 3: Rendering Text to '{RENDERED_TEXT_PNG}' ---")
    if not render_text_to_png_manual_v3(sanskrit_shloka_render, hindi_meaning_render, english_meaning_render, RENDERED_TEXT_PNG, eff_font_size_sk, eff_font_size_hi, eff_font_size_en):
        print(f"Failed to render text to PNG."); return None, None, None
    print(f"Text rendered to '{RENDERED_TEXT_PNG}' successfully.\n")
        
    print(f"--- Step 4: Selecting Background Music ---")
    selected_background_music_file = None
    if BACKGROUND_MUSIC_FILES_LIST:
        available_music_files = [f for f in BACKGROUND_MUSIC_FILES_LIST if os.path.exists(f)]
        if available_music_files:
            selected_background_music_file = random.choice(available_music_files)
            print(f"Randomly selected background music: {selected_background_music_file}")
        else: print("Warning: No BGM files found from the list. Proceeding without music.")
    else: print("BACKGROUND_MUSIC_FILES_LIST empty. Proceeding without music.")

    print(f"--- Step 5: Creating Final Video '{FINAL_OUTPUT_VIDEO}' ---")
    if not create_video_with_ai_bg_overlay_and_music(AI_GENERATED_BACKGROUND_FILENAME, RENDERED_TEXT_PNG, selected_background_music_file, FINAL_OUTPUT_VIDEO):
        print("\nFailed to create final video."); return None, None, None
    
    print(f"\nSuccessfully created video: '{FINAL_OUTPUT_VIDEO}'.")

    # --- Prepare title and description for Facebook ---
    fb_title = sanskrit_shloka_raw.split('\n')[0].strip() # First line of Sanskrit shloka as title
    if len(fb_title) > 100: # Keep title concise
        fb_title = fb_title[:97] + "..."

    fb_description_parts = []
    if sanskrit_shloka_raw.strip():
        fb_description_parts.append(f"Sanskrit Shloka:\n{sanskrit_shloka_raw.strip()}")
    if hindi_meaning_raw.strip():
        fb_description_parts.append(f"Hindi Meaning (हिंदी अर्थ):\n{hindi_meaning_raw.strip()}")
    if english_meaning_raw.strip():
        fb_description_parts.append(f"English Meaning:\n{english_meaning_raw.strip()}")
    
    fb_description_parts.append("\n#Shloka #Sanskrit #Spirituality #Wisdom #SanatanDharma #Hinduism #VedicWisdom #DailyMotivation")
    fb_description = "\n\n".join(fb_description_parts)

    return FINAL_OUTPUT_VIDEO, fb_title, fb_description


# === FACEBOOK UPLOAD MODULE ===
def upload_facebook_reel(video_path_to_upload, title, description, page_id_val, access_token_val):
    """
    Uploads the specified video to Facebook.
    """
    print("\n--- Starting Facebook Upload Process ---")
    if not os.path.exists(video_path_to_upload):
        print(f"❌ Error: Video file for upload not found: {video_path_to_upload}")
        return False
        
    url = f'https://graph.facebook.com/v18.0/{page_id_val}/videos'
    
    try:
        with open(video_path_to_upload, 'rb') as video_file:
            files = {
                'source': video_file,
            }
            data = {
                'title': title,
                'description': description,
                'access_token': access_token_val,
                # 'upload_phase': 'start', # For resumable, not strictly needed for small files
                # 'published': 'true', # Set to true to publish immediately, else it's a draft.
                                     # For Reels, test behavior. Graph API usually uploads then you publish.
                                     # Or it might publish directly if page settings allow.
            }

            print(f"Uploading '{video_path_to_upload}' to Facebook Page ID {page_id_val}...")
            print(f"Title: {title}")
            # print(f"Description: {description}") # Description can be long, optionally print
            
            response = requests.post(url, files=files, data=data)

            if response.status_code == 200:
                response_data = response.json()
                video_id = response_data.get('id')
                print(f"✅ Video uploaded successfully. Video ID: {video_id}")
                print("Response:", response_data)
                return True
            else:
                print("❌ Failed to upload video.")
                print("Status Code:", response.status_code)
                print("Response:", response.text)
                return False
    
    except FileNotFoundError: # Should be caught by the check above, but as a fallback
        print(f"❌ Error: Video file not found at '{video_path_to_upload}'. Cannot upload.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ A network error occurred during upload: {e}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred during upload: {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        print("--- Facebook Upload Process Finished ---")

# === MAIN EXECUTION SCRIPT ===
if __name__ == "__main__":
    print("--- Starting Full Video Creation and Upload Workflow ---")

    # Check for essential files/folders at the start
    if not os.path.exists(FONT_FILE_PATH_SANSKRIT):
        print(f"CRITICAL ERROR: Sanskrit Font file '{FONT_FILE_PATH_SANSKRIT}' not found. Exiting.")
        exit()
    if not os.path.isdir(PREEXISTING_BACKGROUND_IMAGE_FOLDER):
        print(f"CRITICAL ERROR: Background image folder '{PREEXISTING_BACKGROUND_IMAGE_FOLDER}' not found. Exiting.")
        exit()
    
    generated_video_path = None
    fb_video_title = None
    fb_video_description = None
    video_generation_succeeded = False

    for attempt_num in range(1, MAX_OVERALL_ATTEMPTS + 1):
        print(f"\n--- Overall Video Generation Attempt {attempt_num}/{MAX_OVERALL_ATTEMPTS} ---")
        
        video_details = run_video_generation_pipeline() # Returns (path, title, description) or (None,None,None)
        
        if video_details and video_details[0] and os.path.exists(video_details[0]):
            generated_video_path, fb_video_title, fb_video_description = video_details
            video_generation_succeeded = True
            print(f"\n--- Video generation successful on attempt {attempt_num}! ---")
            print(f"Video ready at: {generated_video_path}")
            break
        else:
            print(f"--- Video generation failed on attempt {attempt_num}. ---")
            if attempt_num < MAX_OVERALL_ATTEMPTS:
                print(f"Waiting for {RETRY_WAIT_SECONDS} seconds before retrying...")
                time.sleep(RETRY_WAIT_SECONDS)
            else:
                print("\n--- All video generation attempts failed. ---")

    if video_generation_succeeded and generated_video_path:
        print(f"\n--- Proceeding to upload video: {generated_video_path} ---")
        upload_successful = upload_facebook_reel(
            video_path_to_upload=generated_video_path,
            title=fb_video_title,
            description=fb_video_description,
            page_id_val=FB_PAGE_ID,
            access_token_val=FB_ACCESS_TOKEN
        )

        if upload_successful:
            print(f"\nSUCCESS: Video created and uploaded to Facebook.")
        else:
            print("\nFAILURE: Video was created but upload to Facebook failed.")

        # Optional: Clean up generated files after successful upload or even if upload fails but video was made
        print("\n--- Cleaning up temporary files ---")
        files_to_clean = [RENDERED_TEXT_PNG, AI_GENERATED_BACKGROUND_FILENAME]
        # if upload_successful: # Only delete video if upload was successful
        #     files_to_clean.append(generated_video_path) # Or always delete generated video

        for f_path in files_to_clean:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                    print(f"🗑️ Cleaned up: {f_path}")
                except OSError as e:
                    print(f"⚠️ Warning: Could not clean up file '{f_path}': {e}")
        # To clean the main video:
        # if os.path.exists(generated_video_path):
        #     try:
        #         os.remove(generated_video_path)
        #         print(f"🗑️ Cleaned up video: {generated_video_path}")
        #     except OSError as e:
        #         print(f"⚠️ Warning: Could not clean up video file '{generated_video_path}': {e}")

    else:
        print("\n--- Video generation failed, skipping upload. ---")

    print("\n--- Workflow Finished ---")

# --- END OF COMBINED SCRIPT ---