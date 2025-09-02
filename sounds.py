from gtts import gTTS

# English
tts_mask_en = gTTS("Mask on, you can pass", lang="en")
tts_mask_en.save("mask_on_en.mp3")

tts_nomask_en = gTTS("Please wear your mask", lang="en")
tts_nomask_en.save("no_mask_en.mp3")

# Arabic
tts_mask_ar = gTTS("تم ارتداء الكمامة، يمكنك المرور", lang="ar")
tts_mask_ar.save("mask_on_ar.mp3")

tts_nomask_ar = gTTS("من فضلك ارتدي الكمامة", lang="ar")
tts_nomask_ar.save("no_mask_ar.mp3")
