from flask import Flask, render_template, url_for, request, send_from_directory, redirect
from werkzeug import secure_filename
import os

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['wav', 'mp3'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('thanks',
                                    filename=filename))
    return render_template('index.html')

@app.route('/thanks/<filename>')
def thanks(filename):
    filepath = '/useruploads/' + filename
    context = {'filename': filename, 'filepath': filepath}
    return render_template('thanks.html', context = context)

@app.route('/useruploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
