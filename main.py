from gtts import gTTS
import os

mytext = 'Welcome to geeksforgeeks Joe!'

language = 'en'
myobj = gTTS(text=mytext, lang=language, tld='co.uk', slow=False)
myobj.save("message.mp3")

