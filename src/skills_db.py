"""
Skill vocabulary for the screening system.

Each entry maps a canonical skill name to the surface forms that may appear
in a resume or a job description. Matching is done on whole words only, so
"r" will not match inside "recruiter", and "c++" is escaped safely.

To add a skill: add one line to SKILL_ALIASES. Nothing else needs to change.
"""

SKILL_ALIASES: dict[str, list[str]] = {
    # --- Programming languages ---
    "Python": ["python", "python3"],
    "Java": ["java"],
    "JavaScript": ["javascript", "js", "es6"],
    "TypeScript": ["typescript", "ts"],
    "C": ["c language", "c programming"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "csharp"],
    "Go": ["golang"],
    "Rust": ["rust"],
    "Ruby": ["ruby"],
    "PHP": ["php"],
    "Swift": ["swift"],
    "Kotlin": ["kotlin"],
    "R": ["r programming", "r language"],
    "Scala": ["scala"],
    "MATLAB": ["matlab"],
    "Shell Scripting": ["bash", "shell scripting", "shell script"],

    # --- Data / ML ---
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning", "dl"],
    "Natural Language Processing": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision", "cv", "opencv"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Data Science": ["data science"],
    "Statistics": ["statistics", "statistical analysis"],
    "Scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch", "torch"],
    "Keras": ["keras"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Matplotlib": ["matplotlib"],
    "Seaborn": ["seaborn"],
    "XGBoost": ["xgboost"],
    "NLTK": ["nltk"],
    "spaCy": ["spacy"],
    "Hugging Face": ["hugging face", "huggingface", "transformers"],
    "LangChain": ["langchain"],
    "Large Language Models": ["llm", "llms", "large language model",
                              "large language models", "generative ai", "gen ai"],
    "Prompt Engineering": ["prompt engineering"],
    "Feature Engineering": ["feature engineering"],
    "Model Deployment": ["model deployment", "mlops", "ml ops"],
    "Time Series Analysis": ["time series", "time-series", "forecasting"],
    "Recommendation Systems": ["recommendation system", "recommender system",
                               "recommendation systems", "recommender systems"],

    # --- Web / app ---
    "React": ["react", "react.js", "reactjs"],
    "Angular": ["angular", "angular.js", "angularjs"],
    "Vue": ["vue", "vue.js", "vuejs"],
    "Node.js": ["node.js", "nodejs", "node"],
    "Express.js": ["express.js", "expressjs", "express"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi", "fast api"],
    "Spring Boot": ["spring boot", "springboot", "spring"],
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3"],
    "Tailwind CSS": ["tailwind", "tailwind css"],
    "Bootstrap": ["bootstrap"],
    "REST APIs": ["rest api", "rest apis", "restful", "rest"],
    "GraphQL": ["graphql"],
    "Streamlit": ["streamlit"],
    "Flutter": ["flutter"],
    "Android Development": ["android", "android development"],
    "iOS Development": ["ios", "ios development"],

    # --- Databases ---
    "SQL": ["sql"],
    "MySQL": ["mysql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MongoDB": ["mongodb", "mongo"],
    "SQLite": ["sqlite"],
    "Oracle": ["oracle", "oracle db"],
    "Redis": ["redis"],
    "Elasticsearch": ["elasticsearch", "elastic search"],
    "Firebase": ["firebase"],
    "Data Warehousing": ["data warehouse", "data warehousing", "snowflake",
                         "redshift", "bigquery"],

    # --- Cloud / DevOps ---
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "Google Cloud": ["gcp", "google cloud", "google cloud platform"],
    "Docker": ["docker", "containerization"],
    "Kubernetes": ["kubernetes", "k8s"],
    "CI/CD": ["ci/cd", "cicd", "continuous integration", "jenkins",
              "github actions", "gitlab ci"],
    "Git": ["git", "github", "gitlab", "version control"],
    "Linux": ["linux", "unix", "ubuntu"],
    "Terraform": ["terraform"],
    "Airflow": ["airflow", "apache airflow"],
    "Apache Spark": ["spark", "pyspark", "apache spark"],
    "Hadoop": ["hadoop", "hdfs", "mapreduce"],
    "Kafka": ["kafka", "apache kafka"],

    # --- Tools / BI ---
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Excel": ["excel", "ms excel", "microsoft excel", "advanced excel"],
    "Jupyter": ["jupyter", "jupyter notebook", "google colab", "colab"],
    "JIRA": ["jira"],
    "Figma": ["figma"],
    "Postman": ["postman"],

    # --- Methods / practices ---
    "Agile": ["agile", "scrum", "kanban", "sprint"],
    "Testing": ["unit testing", "pytest", "junit", "selenium",
                "test automation", "qa"],
    "Object Oriented Programming": ["oop", "object oriented", "object-oriented"],
    "Data Structures": ["data structures", "dsa"],
    "Algorithms": ["algorithms", "algorithm design"],
    "System Design": ["system design", "microservices", "distributed systems"],
    "Web Scraping": ["web scraping", "beautifulsoup", "scrapy", "selenium"],

    # --- Soft skills ---
    "Communication": ["communication", "communication skills",
                      "verbal communication", "written communication"],
    "Teamwork": ["teamwork", "team player", "collaboration", "collaborative"],
    "Leadership": ["leadership", "team lead", "mentoring", "mentorship"],
    "Problem Solving": ["problem solving", "problem-solving",
                        "analytical skills", "analytical thinking"],
    "Time Management": ["time management"],
    "Presentation": ["presentation", "presentation skills", "public speaking"],
    "Stakeholder Management": ["stakeholder management", "client management",
                               "client communication"],
}

# Skills that should not dominate a technical match. They still count,
# but at reduced weight (see scoring.SOFT_SKILL_WEIGHT).
SOFT_SKILLS = {
    "Communication", "Teamwork", "Leadership", "Problem Solving",
    "Time Management", "Presentation", "Stakeholder Management",
}


def all_skill_names() -> list[str]:
    """Every canonical skill name, alphabetically sorted."""
    return sorted(SKILL_ALIASES.keys())
