from flask import Flask, render_template, url_for, request, send_from_directory, redirect
from werkzeug import secure_filename
# from audiostuff import save_spectrogram
import os

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['wav', 'mp3'])

application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@application.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('thanks',
                                    filename=filename))
    return render_template('index.html')

@application.route('/thanks/<filename>')
def thanks(filename):
    filepath = os.path.join(application.config['UPLOAD_FOLDER'], filename)
    context = {'filename': filename, 'filepath': filepath}
    save_spectrogram(filepath, filename)
    return render_template('thanks.html', context = context)

@application.route('/talkover/<filename>')
def talkover(filename):
    context = {}
    return render_template('talkover.html', context = context)

@application.route('/useruploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # app.run(host='0.0.0.0', port=port, debug=True)
    application.run(port=port, debug=True)
