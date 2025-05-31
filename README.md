facebook_automator
==================

Overview
--------
facebook_automator is a Python-based tool designed to automate the uploading of videos to Facebook. It streamlines the process by handling video preparation and upload, making it easier for content creators to manage their Facebook video content.

Table of Contents
-----------------
- Features
- Prerequisites
- Installation
- Configuration
  - Facebook Access Token
- Usage
- File Structure
- Troubleshooting
- License
- Acknowledgments

Features
--------
- Automates the uploading of videos to Facebook.
- Utilizes Facebook's Graph API for seamless integration.
- Supports custom video titles and descriptions.
- Logs used content to prevent duplication.

Prerequisites
-------------
Before you begin, ensure you have the following installed:
- Python 3.6 or higher: https://www.python.org/downloads/
- pip (Python package installer)
- Git: https://git-scm.com/downloads

Installation
------------
1. Clone the Repository:
   Open your terminal or command prompt and run:
   git clone https://github.com/RohitRanade/facebook_automator.git

   Navigate to the project directory:
   cd facebook_automator

2. Create a Virtual Environment (Optional but Recommended):
   python -m venv venv

   Activate the virtual environment:
   - Windows:
     venv\Scripts\activate
   - macOS/Linux:
     source venv/bin/activate

3. Install Required Packages:
   pip install -r requirements.txt

   Note: If 'requirements.txt' is not present, you may need to manually install dependencies as you encounter errors.

Configuration
-------------
Facebook Access Token:
To interact with Facebook's API, you'll need an access token. Follow these steps:

1. Create a Facebook Developer Account:
   - Visit https://developers.facebook.com/ and log in with your Facebook account.
   - Click on 'My Apps' > 'Create App'.
   - Choose an app type (e.g., "Business") and provide the necessary details.

2. Set Up Facebook Login:
   - In your app dashboard, navigate to 'Add Product' > 'Facebook Login' > 'Set Up'.
   - Configure the settings as required.

3. Generate Access Token:
   - Go to the Graph API Explorer: https://developers.facebook.com/tools/explorer/
   - Select your app from the top-right dropdown.
   - Click on 'Get Token' > 'Get User Access Token'.
   - In the permissions window, select the necessary permissions (e.g., 'pages_manage_posts', 'pages_read_engagement').
   - Click 'Generate Access Token'.
   - Copy the generated token.

   Note: For long-term use, consider generating a long-lived access token. Refer to Facebook's documentation for more details.

4. Set Environment Variable:
   - Windows:
     set FACEBOOK_ACCESS_TOKEN=your_access_token

   - macOS/Linux:
     export FACEBOOK_ACCESS_TOKEN=your_access_token

   Replace 'your_access_token' with the token you obtained.

   Alternatively, add it to your '.env' file:
   FACEBOOK_ACCESS_TOKEN=your_access_token

Usage
-----
1. Prepare Your Media:
   - Ensure your video file (e.g., 'gemini_video_with_ai_bg_v3.mp4') is present in the project directory.
   - Fonts: Make sure the font files ('NotoSansDevanagari_ExtraCondensed-Medium.ttf', 'NotoSerifDevanagari_ExtraCondensed-Bold.ttf') are in the project directory.

2. Run the Script:
   Execute the main script:
   python AutoUploaderfacebook(PerfectlyWorking).py

   This will upload the specified video to your Facebook page using the provided access token.

File Structure
--------------
- 'AutoUploaderfacebook(PerfectlyWorking).py': Main script that handles video uploading.
- 'gemini_video_with_ai_bg_v3.mp4': Sample video file for upload.
- 'NotoSansDevanagari_ExtraCondensed-Medium.ttf', 'NotoSerifDevanagari_ExtraCondensed-Bold.ttf': Font files used in video overlays.
- 'last_image_index.txt': Tracks the last used image to avoid repetition.
- 'used_shlokas.txt': Records used text overlays to prevent duplicates.

Troubleshooting
---------------
- Missing Modules:
  If you encounter 'ModuleNotFoundError', install the missing module using pip:
  pip install module_name

- Permission Issues:
  Ensure you have the necessary permissions to read/write files in the project directory.

- Invalid Access Token:
  If you receive authentication errors, ensure your Facebook access token is valid and has not expired.

- API Permission Errors:
  Verify that your access token has the required permissions and that your app is in live mode if necessary.

License
-------
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
---------------
Thanks to RohitRanade for creating this project.

Feel free to contribute to this project by submitting issues or pull requests.
