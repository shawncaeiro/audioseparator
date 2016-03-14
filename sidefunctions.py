import librosa
def getsonglength(path_to_audio):
    song, sr = librosa.load(path_to_audio)
    return int(round(len(song) / float(sr)))
