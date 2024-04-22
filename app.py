from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from PyPDF2 import PdfReader
from bcrypt import hashpw, gensalt, checkpw
import os
import io
import jwt
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download("punkt")
# nltk.download("stopwords")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "default-secret-key")
db = SQLAlchemy(app)
migrate = Migrate(app, db)


class PdfText(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    projects = db.relationship("Project", backref="user", lazy=True)


class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_name = db.Column(db.String(255), nullable=False)
    overview = db.Column(db.Text, nullable=False)
    project_status = db.Column(db.String(50), nullable=False)
    proposal_status = db.Column(db.String(50), nullable=False)
    report_status = db.Column(db.String(50), nullable=False)
    proposal_cosine = db.Column(db.String(50), nullable=True)
    proposal_jaccard = db.Column(db.String(50), nullable=True)
    report_cosine = db.Column(db.String(50), nullable=True)
    report_jaccard = db.Column(db.String(50), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)


with app.app_context():
    db.create_all()


def pdf_to_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PdfReader(file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text


def store_text_in_sqlite(text):
    new_text = PdfText(text=text)
    db.session.add(new_text)
    db.session.commit()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/pdfs", methods=["GET"])
def get_all_pdfs():
    all_pdfs = PdfText.query.all()
    pdf_texts = [pdf.text for pdf in all_pdfs]
    return jsonify({"pdf_texts": pdf_texts})


@app.route("/users", methods=["GET"])
def get_all_users():
    all_users = User.query.all()
    user_data = [{"id": user.id, "username": user.username} for user in all_users]
    return jsonify({"users": user_data})


@app.route("/users", methods=["DELETE"])
def delete_all_users():
    User.query.delete()
    db.session.commit()
    return jsonify({"message": "All users deleted successfully"})


@app.route("/pdfs", methods=["DELETE"])
def delete_all_pdfs():
    PdfText.query.delete()
    db.session.commit()
    return jsonify({"message": "All PDFs deleted successfully"})


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        text = pdf_to_text(file_path)
        store_text_in_sqlite(text)

        return (
            jsonify(
                {"message": "File uploaded successfully and data stored in SQLite"}
            ),
            200,
        )

    return jsonify({"error": "Error. Please check or try again"}), 500


@app.route("/check-plagiarism", methods=["POST"])
def check_plagiarism():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # COSINE SIMILARITY CODE
    def preprocess_text(text):
        tokens = word_tokenize(text.lower())

        tokens = [word for word in tokens if word.isalnum() and word not in string.punctuation]

        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        cleaned_text = ' '.join(tokens)

        return cleaned_text

    def calculate_cosine_similarity(doc1, doc2):
        tfidf_vectorizer = TfidfVectorizer()

        tfidf_matrix = tfidf_vectorizer.fit_transform([doc1, doc2])

        cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

        return cosine_sim

    # JACCARD SIMILARITY CODE
    def jaccard_similarity(paragraph1, paragraph2):
        # Tokenize paragraphs into words
        words1 = set(paragraph1.lower().split())
        words2 = set(paragraph2.lower().split())

        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        similarity = intersection / union if union != 0 else 0
        return similarity

    if file:
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        text = pdf_to_text(file_path)

        all_pdfs = PdfText.query.all()
        existing_texts = [pdf.text for pdf in all_pdfs]

        cosine_threshold = 0.4
        jaccard_threshold = 0.5

        for existing_text in existing_texts:
            cleaned_doc1 = preprocess_text(existing_text)
            cleaned_doc2 = preprocess_text(text)

            cosine_sim = calculate_cosine_similarity(cleaned_doc1, cleaned_doc2)
            jaccard_sim = jaccard_similarity(existing_text, text)

            T = "Plagiarism Detected"
            F = "No plagiarism detected"

            if cosine_sim > cosine_threshold and jaccard_sim > jaccard_threshold:
                return (
                    jsonify(
                        {
                            "message": T,
                            "csp": cosine_sim * 100,
                            "jsp": jaccard_sim * 100,
                        }
                    ),
                    200,
                )
            elif cosine_sim < cosine_threshold and jaccard_sim < jaccard_threshold:
                return jsonify({"message": F}), 200
            elif cosine_sim > cosine_threshold and jaccard_sim < jaccard_threshold:
                if jaccard_sim < 0.32:
                    return jsonify({"message": F}), 200
                else:
                    return (
                        jsonify(
                            {
                                "message": T,
                                "csp": cosine_sim * 100,
                                "jsp": jaccard_sim * 100,
                            }
                        ),
                        200,
                    )
            elif cosine_sim < cosine_threshold and jaccard_sim > jaccard_threshold:
                if cosine_sim < 0.2:
                    return jsonify({"message": F}), 200
                else:
                    return (
                        jsonify(
                            {
                                "message": T,
                                "csp": cosine_sim * 100,
                                "jsp": jaccard_sim * 100,
                            }
                        ),
                        200,
                    )

        return jsonify({"message": "No plagiarism detected"}), 200

    return jsonify({"error": "Error. Please check or try again"}), 500


@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if username and password:
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return jsonify({"error": "User already exists"}), 400

        hashed_password = hashpw(password.encode("utf-8"), gensalt())
        new_user = User(username=username, password=hashed_password.decode("utf-8"))

        db.session.add(new_user)
        db.session.commit()

        return jsonify({"message": "User created successfully"}), 201

    return jsonify({"error": "Invalid data"}), 400


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if username and password:
        user = User.query.filter_by(username=username).first()
        if user and checkpw(password.encode("utf-8"), user.password.encode("utf-8")):
            token = jwt.encode(
                {"user_id": user.id, "username": user.username},
                app.config["SECRET_KEY"],
                algorithm="HS256",
            )

            response_data = {
                "message": "Login successful",
                "token": token,
                "user_id": user.id,
            }
            return jsonify(response_data), 200

    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/projects", methods=["POST"])
def create_project():
    data = request.get_json()
    project_name = data.get("project_name")
    overview = data.get("overview")
    project_status = "Ongoing"
    proposal_status = "Pending"
    report_status = "Pending"

    if not all(
        [project_name, overview, project_status, proposal_status, report_status]
    ):
        return jsonify({"error": "Invalid data"}), 400

    user_id = data.get("user_id")

    new_project = Project(
        project_name=project_name,
        overview=overview,
        project_status=project_status,
        proposal_status=proposal_status,
        report_status=report_status,
        user_id=user_id,
    )

    db.session.add(new_project)
    db.session.commit()

    return jsonify({"message": "Project created successfully"}), 201


@app.route("/projects", methods=["GET"])
def get_all_projects():
    all_projects = Project.query.all()
    project_data = [
        {
            "id": project.id,
            "project_name": project.project_name,
            "overview": project.overview,
            "project_status": project.project_status,
            "proposal_status": project.proposal_status,
            "report_status": project.report_status,
            "proposal_cosine": project.proposal_cosine,
            "proposal_jaccard": project.proposal_jaccard,
            "report_jaccard": project.report_cosine,
            "report_cosine": project.report_jaccard,
            "user_id": project.user_id,
        }
        for project in all_projects
    ]
    return jsonify({"projects": project_data})


@app.route("/projects/<int:user_id>", methods=["GET"])
def get_user_projects(user_id):
    user_projects = Project.query.filter_by(user_id=user_id).all()
    project_data = [
        {
            "id": project.id,
            "project_name": project.project_name,
            "overview": project.overview,
            "project_status": project.project_status,
            "proposal_status": project.proposal_status,
            "report_status": project.report_status,
            "proposal_cosine": project.proposal_cosine,
            "proposal_jaccard": project.proposal_jaccard,
            "report_jaccard": project.report_cosine,
            "report_cosine": project.report_jaccard,
            "user_id": project.user_id,
        }
        for project in user_projects
    ]
    return jsonify({"user_projects": project_data})


@app.route("/projects/<int:user_id>/<int:project_id>", methods=["GET"])
def get_project(user_id, project_id):
    project = Project.query.filter_by(user_id=user_id, id=project_id).first()
    if project:
        if project.proposal_cosine:
            project_data = {
                "id": project.id,
                "project_name": project.project_name,
                "overview": project.overview,
                "project_status": project.project_status,
                "proposal_status": project.proposal_status,
                "report_status": project.report_status,
                "proposal_cosine": project.proposal_cosine,
                "proposal_jaccard": project.proposal_jaccard,
                "report_cosine": project.report_cosine,
                "report_jaccard": project.report_jaccard,
                "user_id": project.user_id,
            }
        else:
            project_data = {
                "id": project.id,
                "project_name": project.project_name,
                "overview": project.overview,
                "project_status": project.project_status,
                "proposal_status": project.proposal_status,
                "report_status": project.report_status,
                "proposal_cosine": "",
                "proposal_jaccard": "",
                "report_cosine": "",
                "report_jaccard": "",
                "user_id": project.user_id,
            }
        return jsonify({"project": project_data})
    else:
        return jsonify({"message": "Project not found"}), 404


@app.route("/projects/<int:user_id>/<int:project_id>", methods=["DELETE"])
def delete_project(user_id, project_id):
    project = Project.query.filter_by(user_id=user_id, id=project_id).first()
    if project:
        db.session.delete(project)
        db.session.commit()
        return jsonify({"message": "Project deleted successfully"}), 200
    else:
        return jsonify({"message": "Project not found"}), 404


@app.route("/projects/<int:user_id>/<int:project_id>", methods=["PUT"])
def update_project(user_id, project_id):
    project = Project.query.filter_by(user_id=user_id, id=project_id).first()
    if project:
        data = request.form
        project.project_name = data.get("project_name", project.project_name)
        project.project_status = data.get("project_status", project.project_status)
        project.proposal_status = data.get("proposal_status", project.proposal_status)
        project.report_status = data.get("report_status", project.report_status)
        project.overview = data.get("overview", project.overview)

        if "report_cosine" in data:
            project.report_cosine = data["report_cosine"]
        if "report_jaccard" in data:
            project.report_jaccard = data["report_jaccard"]
        if "proposal_cosine" in data:
            project.proposal_cosine = data["proposal_cosine"]
        if "proposal_jaccard" in data:
            project.proposal_jaccard = data["proposal_jaccard"]

        db.session.commit()
        project_data = {
            "id": project.id,
            "user_id": project.user_id,
            "project_name": project.project_name,
            "project_status": project.project_status,
            "proposal_status": project.proposal_status,
            "report_status": project.report_status,
            "overview": project.overview,
            "report_cosine": project.report_cosine,
            "report_jaccard": project.report_jaccard,
            "proposal_cosine": project.proposal_cosine,
            "proposal_jaccard": project.proposal_jaccard,
        }
        return (
            jsonify(
                {"message": "Project updated successfully", "project": project_data}
            ),
            200,
        )
    else:
        return jsonify({"message": "Project not found"}), 404
