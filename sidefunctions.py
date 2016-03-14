import librosa
def getsonglength(path_to_audio):
    song, sr = librosa.load(path_to_audio)
    return len(song) / float(sr)
