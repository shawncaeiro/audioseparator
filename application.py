from flask import Flask, render_template, url_for, request, send_from_directory, redirect
from werkzeug import secure_filename
from sidefunctions import getsonglength, combinesongs
import logging
import os
# from flask_sslify import SSLify

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['wav'])

application = Flask(__name__)
# application.debug = False
# if 'DYNO' in os.environ: # only trigger SSLify if the app is running on Heroku
#     sslify = SSLify(application)

application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@application.route('/', methods=['GET', 'POST'])
def index():
    application.logger.warning("INDEX!")
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('thanks',
                                    filename=filename, _scheme='https', _external=True))
    return render_template('index.html')

@application.route('/uploadtalk/<filename>', methods=['POST'])
def uploadtalk(filename):
    application.logger.warning("uploadtalk!")
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # application.logger.warning(file.filename)
            # filename = secure_filename(file.filename)
            file.save(os.path.join(application.config['UPLOAD_FOLDER'], 'voicey' + filename))
            combinedpath = os.path.join(application.config['UPLOAD_FOLDER'], 'combined' + filename)
            combinesongs(os.path.join(application.config['UPLOAD_FOLDER'], filename), os.path.join(application.config['UPLOAD_FOLDER'], 'voicey' + filename), combinedpath)
    return url_for('uploaded_file', filename = 'combined' + filename)

@application.route('/thanks/<filename>')
def thanks(filename):
    filepath = os.path.join(application.config['UPLOAD_FOLDER'], filename)
    context = {'filename': filename, 'filepath': filepath}
    # save_spectrogram(filepath, filename)
    return render_template('thanks.html', context = context)

@application.route('/talkover/<filename>')
def talkover(filename):
    uploadurl = '/uploadtalk/' + filename
    truelength = getsonglength(os.path.join(application.config['UPLOAD_FOLDER'], filename))
    length = int(round(truelength))
    lengthms = int(round(truelength * 1000))
    application.logger.warning(length)
    context = {'length':length, 'lengthms':lengthms, 'uploadurl':uploadurl}
    return render_template('talkover.html', context = context)

@application.route('/useruploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(application.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    application.run(host='0.0.0.0', port=port, debug=True)
    # application.run(port=port, debug=True)
