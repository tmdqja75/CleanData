import json
import os
import uuid

import pandas as pd
import werkzeug
from flask import Flask, request, send_file, send_from_directory, abort, current_app
from flask_restful import Api, Resource, reqparse
from werkzeug.utils import secure_filename

from main import run

app = Flask(__name__)
app.secret_key = "secret key"

api = Api(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# Make directory if "uploads" folder not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DOWNLOAD_FOLDER"] = '/Users/seungbeomha/AI_MLStudy/Projects/FinalProject/CleanData/src'
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "gif", "mov", "mp4", "mpeg"])
# parse = reqparse.RequestParser()
# parse.add_argument(
#     "file",
#     werkzeug.datastructures.FileStorage,
#     location="files",
#     required=True,
#     action="append",
# )


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def _get_files():
    file_list = os.path.join(UPLOAD_FOLDER, 'files.json')
    if os.path.exists(file_list):
        with open(file_list) as fh:
            return json.load(fh)
    return {}

@api.representation('application/octet-stream')
def output_file(data, code, headers):
    filepath = os.path.join(data["directory"], data["filename"])

    response = send_file(
        filename_or_fp=filepath,
        mimetype="application/octet-stream",
        as_attachment=True,
        attachment_filename=data["filename"]
    )
    return response


class Upload(Resource):
    # def get(self, user_id):
    #     return {'data': 'hello world!'}
    def post(self, user_id):

        if "file" not in request.files:
            return {"message": f"No file part:{user_id}"}

        app.logger.info(request.files)
        files = request.files.getlist("file")
        app.logger.info(files)

        if not files:
            return {"message": f"File list empty:{user_id}"}

        print(app.config["UPLOAD_FOLDER"])
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{user_id}/")
        if not os.path.isdir(file_path):
            os.mkdir(file_path)

        for file in files:
            if file and allowed_file(file.filename):
                original_filename = file.filename
                extension = original_filename.rsplit(".", 1)[1].lower()
                filename = str(uuid.uuid1()) + "." + extension
                # filename = secure_filename(file.filename)
                file_path = os.path.join(
                    app.config["UPLOAD_FOLDER"], f"{user_id}/", filename
                )
                file.save(file_path)
                file_list = os.path.join(UPLOAD_FOLDER, "files.json")
                files = _get_files()
                files[filename] = original_filename

                with open(file_list, "w") as fh:
                    json.dump(files, fh)

        return {"message": f"File(s) successfully uploaded on {user_id}"}


# class Process(Resource):
#     def get(self, filename):
        
#         try:
#             print(current_app.root_path)
#             return send_from_directory(app.config["DOWNLOAD_FOLDER"], filename=filename, as_attachment=False)
#         except FileNotFoundError:
#             abort(404)
        

#         # # df_result = run()
#         # df_result = pd.DataFrame()
#         # df_result.to_csv('./answer.csv', index=False, encoding='UTF8')
#         # return {"directory": "./",
#         #         'filename': filename}

@app.route('/process/<filename>')
def send_csv(filename):
    return send_from_directory('result', filename)

api.add_resource(Upload, "/user/<string:user_id>")
# api.add_resource(Process, "/process/<path:filename>")

if __name__ == "__main__":
    app.run(port=5001, debug=True)
# 서버에서 돌릴 시
# if __name__=="__main__":
#    app.run(host='0.0.0.0', port=8080)
