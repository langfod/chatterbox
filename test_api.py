import requests


test_wav_file = "assets\\dlc1seranavoice.wav"
speaker_name = "dlc1seranavoice"
language = "en"
# load test wav file
# create POST payload
#  speaker_name: str = Form(...),
#   language: str = Form("en"),
#    wav_file: UploadFile = File(...)
# send to http://localhost:7680/create_and_store_latents

files = {
    "wav_file": open(test_wav_file, "rb")
}
data = {
    "speaker_name": speaker_name,
    "language": language
}

response = requests.post("http://localhost:7680/create_and_store_latents", files=files, data=data)
print(response.json())      



#  text: str
#    speaker_wav: Optional[str] = None
#    language: Optional[str] = "en"
#    accent: Optional[str] = None
#    save_path: Optional[str] = None
# create POST payload
# send to http://localhost:7680/tts_to_audio/
# save response to save_path
test_text = "Hello, this is a test of the text to speech synthesis."
speaker_wav = "dlc1seranavoice"
language = "en"
save_path = "output.wav"

data = {
    "text": test_text,
    "speaker_wav": speaker_wav,
    "language": language,
    "save_path": save_path
}

response = requests.post("http://localhost:7680/tts_to_audio/", data=data)
with open(save_path, "wb") as f:
    f.write(response.content)