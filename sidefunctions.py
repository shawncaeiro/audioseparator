import librosa
def getsonglength(path_to_audio):
    song, sr = librosa.load(path_to_audio)
    return len(song) / float(sr)

def combinesongs(path_to_audio, path_to_voice, path_of_output):
    song, sr = librosa.load(path_to_audio)
    voice, sr = librosa.load(path_to_voice)
    song[0:len(voice)] += voice
    librosa.write_wav(path_of_output, song, sr)
