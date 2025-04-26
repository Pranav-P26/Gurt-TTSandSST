from faster_whisper import WhisperModel
import pyaudio
import wave

def main():
    record()
    speechToText()

def record():
    #specs of the recording
    chunk = 1024
    sample_format = pyaudio.paInt16  
    channels = 2
    fs = 44100 
    # how long it will record for
    seconds = 5
    
    #output
    filename = "output.wav"
    
    #uses portaudio to record
    p = pyaudio.PyAudio()  

    print('Recording')
    # streams the mic input
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)
    # frame array
    frames = []  

    # store data in chunks for 5 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # stop and close the stream 
    stream.stop_stream()
    stream.close()
    # Terminate the portaudio
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    
def speechToText():
    #smallest - biggest: tiny, base, small, idk the rest but dont use them (.en makes it english only)
    model_size = "base.en"

    # model for using CPU (other compute types do NOT WORK)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe("output.wav", beam_size=8)

    text = ""

    for segment in segments:
        text += segment.text
        
    print(text)
    
# so that we could test one function at a time
if __name__ == "__main__":
    main()
