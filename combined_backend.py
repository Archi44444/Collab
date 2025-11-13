import io
import json
import sqlite3
import base64
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import Counter

# Silence KMeans warning
import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on LangChain")

# ===========================
# CONFIGURATION & INITIALIZATION
# ===========================
DB_PATH = "peer_review.db"

# Hugging Face Models
print("🔄 Loading AI/ML Models...")

# 1. For Project Analytics (backend_analytics.py)
# Sentiment Analysis
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)
# Text Summarization
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1
)
# Zero-shot Classification for Risk Assessment
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1
)

# 2. For Teammate Recommender (backend_api.py) & Peer Review (backend_peer.py)
# Sentence Transformer for Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)

# 3. For Peer Review Sentiment (re-using the one from analytics or explicitly from peer file)
sentiment_pipe = sentiment_analyzer # Use the same pipeline instance

print("✅ Models loaded successfully!")

# FASTAPI SETUP
app = FastAPI(
    title="Unified AI Backend: Analytics, Recommender, & Peer Review",
    description="A single API combining project analytics, teammate recommendation, and peer review systems.",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# DATA MODELS (Combined)
# ===========================

# From backend_analytics.py
class StudentData(BaseModel):
    student: str
    tasks_assigned: int
    tasks_completed: int
    commits: int
    feedback: str
    pull_requests: Optional[int] = 0
    code_reviews: Optional[int] = 0
    bugs_fixed: Optional[int] = 0
    response_time_hours: Optional[float] = 0

class ProjectUpdate(BaseModel):
    student: str
    update_text: str
    date: Optional[str] = None

class TeamData(BaseModel):
    students: List[StudentData]
    project_deadline: Optional[str] = None

# From backend_peer.py
class TeammateIn(BaseModel):
    name: str
    role: Optional[str] = "Team Member"
    skills: Optional[Dict[str, Any]] = {}

class ProjectCreate(BaseModel):
    title: str
    description: Optional[str] = ""

class ReviewIn(BaseModel):
    reviewer: str
    reviewee: str
    rating: float
    comment: str
    skills: Dict[str, int]
    date: Optional[str] = None

class AutoAssignRequest(BaseModel):
    project_id: int
    top_k: int = 2

# From backend_api.py / Initialize
class StudentProfile(BaseModel):
    id: int
    name: str
    year: str
    skills: List[str]
    bio: str
    department: str
    projectsCompleted: int
    rating: float
    availability: str
    avatar: str

class InitializeRequest(BaseModel):
    students: List[StudentProfile]

# ===========================
# IN-MEMORY DATA STORAGE (for Analytics and Recommender)
# ===========================
project_data_store = [] # for backend_analytics.py
updates_store = [] # for backend_analytics.py

# ===========================
# UTILITY FUNCTIONS (Combined)
# ===========================

# --- DB Helpers (from backend_peer.py) ---
def get_conn():
    return sqlite3.connect(DB_PATH)

def ndarray_to_blob(arr: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.save(bio, arr, allow_pickle=False)
    bio.seek(0)
    return bio.read()

def blob_to_ndarray(blob: bytes) -> np.ndarray:
    bio = io.BytesIO(blob)
    bio.seek(0)
    return np.load(bio, allow_pickle=False)

def teammate_embedding_from_skills(skills: Dict[str, Any]) -> np.ndarray:
    text = " ".join([f"{k}:{v}" for k, v in (skills or {}).items()])
    if text.strip() == "":
        text = "general"
    emb = embedder.encode(text, normalize_embeddings=True)
    return emb

def init_db():
    conn = get_conn()
    c = conn.cursor()
    # projects
    c.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        description TEXT,
        created_at TEXT
    )""")
    # teammates (simple schema)
    c.execute("""
    CREATE TABLE IF NOT EXISTS teammates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER,
        name TEXT,
        role TEXT,
        skills TEXT,   -- JSON string
        embedding BLOB,
        FOREIGN KEY(project_id) REFERENCES projects(id)
    )""")
    # reviews
    c.execute("""
    CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER,
        reviewer TEXT,   -- name or id string
        reviewee TEXT,
        rating REAL,
        date TEXT,
        comment TEXT,
        skills_json TEXT,
        sentiment_label TEXT,
        sentiment_score REAL,
        FOREIGN KEY(project_id) REFERENCES projects(id)
    )""")
    # assignments (for smart reviewer assignment)
    c.execute("""
    CREATE TABLE IF NOT EXISTS assignments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER,
        reviewer_name TEXT,
        reviewee_name TEXT,
        assigned_at TEXT,
        status TEXT DEFAULT 'assigned'
    )""")
    conn.commit()
    conn.close()

init_db()

# --- AI Analytics Helpers (from backend_analytics.py) ---

def analyze_sentiment_detailed(text: str) -> Dict[str, Any]:
    """Enhanced sentiment analysis with confidence scores"""
    result = sentiment_analyzer(text[:512])[0]
    
    return {
        "sentiment": result['label'],
        "confidence": round(result['score'], 3),
        "sentiment_emoji": "😊" if result['label'] == "POSITIVE" else "😟"
    }

def classify_risk_level(student_data: Dict) -> Dict[str, Any]:
    """AI-powered risk classification using zero-shot learning"""
    
    description = f"""
    Student {student_data['student']} has completed {student_data['tasks_completed']} 
    out of {student_data['tasks_assigned']} tasks with {student_data['commits']} commits.
    Feedback: {student_data['feedback'][:200]}
    """
    
    candidate_labels = ["high performer", "at risk", "needs support", "on track"]
    
    result = classifier(description, candidate_labels)
    
    top_label = result['labels'][0]
    confidence = result['scores'][0]
    
    risk_mapping = {
        "high performer": ("🟢 Excellent", "green"),
        "on track": ("🟢 On Track", "green"),
        "needs support": ("🟠 Needs Support", "orange"),
        "at risk": ("🔴 At Risk", "red")
    }
    
    status, color = risk_mapping.get(top_label, ("🟠 Moderate", "orange"))
    
    return {
        "status": status,
        "color": color,
        "ai_classification": top_label,
        "confidence": round(confidence, 3)
    }

def calculate_performance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive performance metrics"""
    
    df['completion_rate'] = (df['tasks_completed'] / df['tasks_assigned'] * 100).round(2)
    
    df['engagement_score'] = (
        df['commits'] * 0.3 + 
        df['pull_requests'] * 0.25 + 
        df['code_reviews'] * 0.25 + 
        df['bugs_fixed'] * 0.2
    ).round(2)
    
    if df['engagement_score'].max() > 0:
        df['engagement_score'] = (
            df['engagement_score'] / df['engagement_score'].max() * 100
        ).fillna(0).round(2)
    
    df['quality_score'] = (
        (df['completion_rate'] * 0.4) +
        (df['engagement_score'] * 0.3) +
        (df['sentiment_score'] * 100 * 0.3)
    ).round(2)
    
    df['velocity'] = (df['tasks_completed'] / df['commits'].replace(0, 1)).round(2)
    
    if df['response_time_hours'].max() > 0:
        df['responsiveness_score'] = (
            100 - (df['response_time_hours'] / df['response_time_hours'].max() * 100)
        ).fillna(0).round(2)
    else:
        df['responsiveness_score'] = 100.00
    
    return df

def generate_ai_recommendations(student_data: Dict) -> List[str]:
    """Generate personalized recommendations using AI"""
    
    recommendations = []
    
    if student_data['completion_rate'] < 60:
        recommendations.append("⚠️ Priority: Focus on completing pending tasks. Consider breaking tasks into smaller chunks.")
    elif student_data['completion_rate'] < 80:
        recommendations.append("📈 Improve task completion rate by 20% to reach optimal performance.")
    else:
        recommendations.append("✅ Excellent task completion! Maintain this momentum.")
    
    if student_data['engagement_score'] < 40:
        recommendations.append("🤝 Increase participation: More code reviews and PRs will boost team collaboration.")
    elif student_data['engagement_score'] < 70:
        recommendations.append("💡 Good engagement! Try mentoring others to further enhance team dynamics.")
    
    if student_data['sentiment'] == "NEGATIVE":
        recommendations.append("🆘 Action needed: Schedule a 1-on-1 to address concerns and provide support.")
    
    if student_data['commits'] < 10:
        recommendations.append("💻 Increase commit frequency for better progress tracking and collaboration.")
    
    if student_data.get('response_time_hours', 0) > 48:
        recommendations.append("⏱️ Improve response time to enhance team communication.")
    
    return recommendations

def summarize_feedback(feedback_list: List[str]) -> str:
    """Summarize multiple feedback entries using AI"""
    
    combined_feedback = " ".join(feedback_list)
    
    if len(combined_feedback) > 100:
        summary = summarizer(
            combined_feedback[:1024],
            max_length=100,
            min_length=30,
            do_sample=False
        )
        return summary[0]['summary_text']
    
    return combined_feedback

# --- Project Analytics (from backend_peer.py) ---

def project_analytics(project_id: int):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id,title,description,created_at FROM projects WHERE id = ?", (project_id,))
    proj = c.fetchone()
    if not proj:
        conn.close()
        raise HTTPException(404, "Project not found")
    # reviews
    c.execute("SELECT rating, skills_json FROM reviews WHERE project_id = ?", (project_id,))
    rows = c.fetchall()
    conn.close()
    if not rows:
        return {"totalReviews": 0, "avgRating": 0, "completionRate": 0, "topSkill": "N/A"}

    ratings = [r[0] for r in rows]
    avg_rating = sum(ratings) / len(ratings)
    # aggregate skills
    skill_sums = {}
    for _, skills_json in rows:
        skills = json.loads(skills_json) if skills_json else {}
        for k, v in skills.items():
            skill_sums[k] = skill_sums.get(k, 0) + v
    # compute averages
    skill_avgs = {k: skill_sums[k] / len(rows) for k in skill_sums}
    if skill_avgs:
        top_skill = max(skill_avgs.items(), key=lambda x: x[1])[0]
        top_skill_name = " ".join(w.capitalize() for w in top_skill.split())
    else:
        top_skill_name = "N/A"
    return {
        "totalReviews": len(rows),
        "avgRating": round(avg_rating, 1),
        "completionRate": 100,
        "topSkill": top_skill_name
    }

# ===========================
# AI TEAMMATE RECOMMENDER CLASS (from backend_api.py, adapted)
# ===========================

class AITeammateRecommender:
    """AI-Powered Teammate Recommendation Engine"""
    
    def __init__(self):
        self.model = embedder # Use the global SentenceTransformer instance
        self.df: pd.DataFrame = None
        self.embeddings = None
        self.similarity_matrix = None
        self.clusters = None
        
    def load_data(self, data: List[Dict]):
        """Load and prepare student data"""
        self.df = pd.DataFrame(data)
        self._prepare_profiles()
        self._generate_embeddings()
        self._compute_similarity()
        self._perform_clustering()
        
    def _prepare_profiles(self):
        """Create profile text from student data"""
        profile_parts = []
        
        for _, row in self.df.iterrows():
            parts = []
            if 'skills' in row and row['skills']:
                # Ensure skills are a list of strings
                if isinstance(row['skills'], str):
                    row['skills'] = [s.strip() for s in row['skills'].split(',')]
                parts.append(' '.join(row['skills']))
            if 'department' in row:
                parts.append(str(row['department']))
            if 'bio' in row:
                parts.append(str(row['bio']))
            if 'year' in row:
                parts.append(str(row['year']))
            
            profile_parts.append(' '.join(parts))
        
        self.df['profile_text'] = profile_parts
    
    def _generate_embeddings(self):
       try:
           print("⚙️ Generating embeddings...")
           self.embeddings = self.model.encode(
                self.df['profile_text'].tolist(),
                show_progress_bar=False,
                batch_size=32
            )
           print("✅ Embeddings generated successfully!")

       except Exception as e:
         print(f"⚠️ Embedding generation failed: {e}")
         self.embeddings = np.zeros((len(self.df), self.model.get_sentence_embedding_dimension()))


    def _compute_similarity(self):
        """Compute similarity matrix"""
        self.similarity_matrix = cosine_similarity(self.embeddings)
    
    def _perform_clustering(self, n_clusters=4):
        """Cluster students into groups"""
        n_clusters = min(n_clusters, len(self.df))
        if n_clusters < 2:
             self.df['cluster'] = 0
             self.clusters = None
             return

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = kmeans.fit_predict(self.embeddings)
        self.clusters = kmeans
    
    def get_recommendations(self, user_id, top_n=8, diversity_weight=0.3, filters=None):
        if self.df is None: return []

        if user_id not in self.df['id'].values:
            return []
        
        idx = self.df.index[self.df['id'] == user_id].tolist()[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Apply diversity penalty
        if diversity_weight > 0 and 'cluster' in self.df.columns:
            user_cluster = self.df.loc[idx, 'cluster']
            adjusted_scores = []
            for i, score in sim_scores:
                if i == idx:
                    adjusted_scores.append((i, 0))
                else:
                    cluster_penalty = 1 - diversity_weight if self.df.loc[i, 'cluster'] == user_cluster else 1
                    adjusted_scores.append((i, score * cluster_penalty))
            sim_scores = adjusted_scores
        
        # Apply filters
        filtered_indices = []
        if filters:
            for i, score in sim_scores:
                if i == idx: continue
                    
                student = self.df.iloc[i]
                
                if filters.get('skills') and len(filters['skills']) > 0:
                    student_skills = student.get('skills', [])
                    if isinstance(student_skills, str): # Handle case where skills might be string
                        student_skills = [s.strip() for s in student_skills.split(',')]
                    if not any(skill.lower() in [s.lower() for s in student_skills] for skill in filters['skills']):
                        continue
                
                if filters.get('years') and len(filters['years']) > 0:
                    if student.get('year') not in filters['years']:
                        continue
                
                if filters.get('departments') and len(filters['departments']) > 0:
                    if student.get('department') not in filters['departments']:
                        continue
                
                filtered_indices.append((i, score))
            
            # If filters are applied, use the filtered list, otherwise use the original scores
            sim_scores = filtered_indices if filtered_indices else sim_scores
        
        # Sort by similarity, filter out self (index 0 might be self)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_matches = [(i, score) for i, score in sim_scores if i != idx][:top_n]
        
        # Build recommendations
        recommendations = []
        for i, score in top_matches:
            student = self.df.iloc[i].to_dict()
            student['match_score'] = float(score)
            student['match_percentage'] = int(score * 100)
            student['cluster_id'] = int(student.get('cluster', 0)) if 'cluster' in student else 0
            recommendations.append(student)
        
        return recommendations
    
    def get_smart_recommendations(self, user_id, user_query="", top_n=8):
        if self.df is None or not user_query: 
            return self.get_recommendations(user_id, top_n=top_n)

        # Encode the query
        query_embedding = self.model.encode([user_query])[0]
        
        # Find students matching the query
        query_similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Combine with user similarity
        combined_scores = query_similarities
        if user_id in self.df['id'].values:
            idx = self.df.index[self.df['id'] == user_id].tolist()[0]
            user_similarities = self.similarity_matrix[idx]
            
            # Weighted combination
            combined_scores = 0.6 * query_similarities + 0.4 * user_similarities
        
        # Get top matches
        top_indices = np.argsort(combined_scores)[::-1]
        
        recommendations = []
        for i in top_indices:
            if self.df.iloc[i]['id'] == user_id:
                continue
                
            student = self.df.iloc[i].to_dict()
            student['match_score'] = float(combined_scores[i])
            student['match_percentage'] = int(combined_scores[i] * 100)
            recommendations.append(student)
        
        return recommendations[:top_n]
    
    def form_optimal_team(self, project_description, team_size=4, exclude_ids=None):
        if self.df is None: return []

        # Encode project description
        project_embedding = self.model.encode([project_description])[0]
        
        # Calculate relevance scores
        relevance_scores = cosine_similarity([project_embedding], self.embeddings)[0]
        
        team = []
        used_clusters = set()
        exclude_ids = exclude_ids or []
        
        # Sort by relevance
        sorted_indices = np.argsort(relevance_scores)[::-1]
        
        for idx in sorted_indices:
            if len(team) >= team_size:
                break
            
            student_id = self.df.iloc[idx]['id']
            if student_id in exclude_ids:
                continue
            
            cluster = self.df.iloc[idx]['cluster'] if 'cluster' in self.df.columns else 0
            
            # Prioritize diversity (simple rule: pick one from each cluster first)
            if cluster not in used_clusters or len(used_clusters) >= team_size:
                team.append({
                    **self.df.iloc[idx].to_dict(),
                    'relevance_score': float(relevance_scores[idx]),
                    'match_percentage': int(relevance_scores[idx] * 100)
                })
                used_clusters.add(cluster)
        
        return team
    
    def get_analytics(self):
        if self.df is None:
            return {}
        
        # Calculate average similarities
        avg_similarities = []
        if self.similarity_matrix is not None:
            for i in range(len(self.similarity_matrix)):
                # Ensure no division by zero or empty array
                sims = [self.similarity_matrix[i][j] for j in range(len(self.similarity_matrix)) if i != j]
                avg_sim = np.mean(sims) if sims else 0
                avg_similarities.append(avg_sim)
        
        # Cluster Distribution
        cluster_dist = self.df['cluster'].value_counts().to_dict() if 'cluster' in self.df.columns else {}
        
        return {
            'total_students': len(self.df),
            'num_clusters': len(self.df['cluster'].unique()) if 'cluster' in self.df.columns else 0,
            'avg_similarity': float(np.mean(avg_similarities)) if avg_similarities else 0.0,
            'cluster_distribution': {int(k): int(v) for k,v in cluster_dist.items()},
            'top_skills': self._get_top_skills(),
            'department_distribution': self.df['department'].value_counts().to_dict() if 'department' in self.df.columns else {}
        }
    
    def _get_top_skills(self):
        """Get top skills across all students"""
        all_skills = []
        if 'skills' in self.df.columns:
            for skills in self.df['skills']:
                if isinstance(skills, list):
                    all_skills.extend(skills)
                elif isinstance(skills, str):
                    all_skills.extend([s.strip() for s in skills.split(',')])
        
        skill_counts = Counter(all_skills)
        return dict(skill_counts.most_common(10))

# Initialize recommender instance
recommender = AITeammateRecommender()

# ===========================
# API ENDPOINTS (FastAPI)
# ===========================

@app.get("/")
async def root():
    """Root endpoint for API info"""
    return {
        "message": "Unified AI Backend API",
        "version": "2.0.0",
        "services": ["Project Analytics", "Teammate Recommender", "Peer Review System"],
        "docs": "/docs"
    }

# --- Teammate Recommender Endpoints (from backend_api.py) ---

@app.post("/api/recommender/initialize", status_code=200)
async def initialize_system(payload: InitializeRequest):
    """Initialize the recommendation system with student data"""
    try:
        data = [s.model_dump() for s in payload.students]
        
        if not data:
            raise HTTPException(status_code=400, detail='No student data provided')
        
        recommender.load_data(data)
        
        return {
            'success': True,
            'message': 'System initialized successfully',
            'stats': recommender.get_analytics()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Initialization error: {str(e)}')

@app.post("/api/recommender/recommendations/{user_id}", status_code=200)
async def get_recommendations_recommender(user_id: int, top_n: int = Query(8), diversity_weight: float = Query(0.3), filters: Optional[str] = Query(None)):
    """Get AI-powered recommendations for a user"""
    try:
        filter_dict = json.loads(filters) if filters else {}
        
        recommendations = recommender.get_recommendations(
            user_id, 
            top_n=top_n,
            diversity_weight=diversity_weight,
            filters=filter_dict
        )
        
        return {
            'success': True,
            'recommendations': recommendations,
            'total': len(recommendations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Recommendation error: {str(e)}')

@app.post("/api/recommender/smart-search", status_code=200)
async def smart_search(data: Dict):
    """Smart search with AI-powered query understanding"""
    try:
        user_id = data.get('user_id')
        query = data.get('query', '')
        top_n = data.get('top_n', 8)
        
        recommendations = recommender.get_smart_recommendations(
            user_id,
            user_query=query,
            top_n=top_n
        )
        
        return {
            'success': True,
            'recommendations': recommendations,
            'query': query
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Smart search error: {str(e)}')

@app.post("/api/recommender/form-team", status_code=200)
async def form_team(data: Dict):
    """Form optimal team for a project"""
    try:
        project_description = data.get('project_description', '')
        team_size = data.get('team_size', 4)
        exclude_ids = data.get('exclude_ids', [])
        
        team = recommender.form_optimal_team(
            project_description,
            team_size=team_size,
            exclude_ids=exclude_ids
        )
        
        return {
            'success': True,
            'team': team,
            'team_size': len(team)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Form team error: {str(e)}')

@app.get("/api/recommender/analytics", status_code=200)
async def get_analytics_recommender():
    """Get system analytics"""
    try:
        analytics = recommender.get_analytics()
        return {
            'success': True,
            'analytics': analytics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Analytics error: {str(e)}')

@app.get("/api/recommender/similarity/{user_id1}/{user_id2}", status_code=200)
async def get_similarity(user_id1: int, user_id2: int):
    """Get similarity score between two users"""
    try:
        if recommender.df is None:
            raise HTTPException(status_code=400, detail='System not initialized')
        
        df = recommender.df
        if user_id1 not in df['id'].values or user_id2 not in df['id'].values:
            raise HTTPException(status_code=404, detail='One or both user IDs not found')

        idx1 = df.index[df['id'] == user_id1].tolist()[0]
        idx2 = df.index[df['id'] == user_id2].tolist()[0]
        
        similarity = float(recommender.similarity_matrix[idx1][idx2])
        
        return {
            'success': True,
            'similarity': similarity,
            'match_percentage': int(similarity * 100)
        }
    except HTTPException:
        raise # Re-raise known HTTPException
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Similarity error: {str(e)}')


# --- Project Analytics Endpoints (from backend_analytics.py) ---

@app.post("/api/analytics")
async def analyze_project(team_data: TeamData):
    """Main analytics endpoint with AI-powered insights"""
    
    try:
        data_dict = {k: [getattr(s, k) for s in team_data.students] for k in StudentData.model_fields.keys()}
        df = pd.DataFrame(data_dict)
        
        global project_data_store
        project_data_store = data_dict
        
        # Sentiment Analysis
        sentiment_results = df['feedback'].apply(analyze_sentiment_detailed)
        df['sentiment'] = sentiment_results.apply(lambda x: x['sentiment'])
        df['sentiment_score'] = sentiment_results.apply(lambda x: x['confidence'])
        df['sentiment_emoji'] = sentiment_results.apply(lambda x: x['sentiment_emoji'])
        
        # Calculate performance metrics
        df = calculate_performance_metrics(df)
        
        # AI Risk Classification
        risk_results = []
        for _, row in df.iterrows():
            risk_info = classify_risk_level(row.to_dict())
            risk_results.append(risk_info)
        
        df['project_status'] = [r['status'] for r in risk_results]
        df['risk_color'] = [r['color'] for r in risk_results]
        df['ai_confidence'] = [r['confidence'] for r in risk_results]
        
        # Generate recommendations for each student
        recommendations_dict = {}
        for _, row in df.iterrows():
            recommendations_dict[row['student']] = generate_ai_recommendations(row.to_dict())
        
        # Team-level analytics
        team_stats = {
            "total_students": len(df),
            "average_completion_rate": round(df['completion_rate'].mean(), 2),
            "average_quality_score": round(df['quality_score'].mean(), 2),
            "total_commits": int(df['commits'].sum()),
            "total_tasks_completed": int(df['tasks_completed'].sum()),
            "positive_sentiment_count": int((df['sentiment'] == 'POSITIVE').sum()),
            "at_risk_count": int(df['project_status'].str.contains('Risk').sum()),
            "high_performers": df[df['quality_score'] > 80]['student'].tolist(),
            "needs_attention": df[df['quality_score'] < 60]['student'].tolist()
        }
        
        students_data = df.to_dict('records')
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "team_statistics": team_stats,
            "students": students_data,
            "recommendations": recommendations_dict,
            "ai_models_used": [
                "distilbert-sentiment-analysis",
                "bart-zero-shot-classification"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/api/analytics/sentiment")
async def analyze_sentiment_endpoint(text: str):
    """Standalone sentiment analysis endpoint"""
    try:
        result = analyze_sentiment_detailed(text)
        return {
            "success": True,
            "text": text[:100] + "..." if len(text) > 100 else text,
            "analysis": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis error: {str(e)}")

@app.get("/api/analytics/student/{student_name}")
async def get_student_details(student_name: str):
    """Get detailed analytics for a specific student"""
    if not project_data_store:
        raise HTTPException(status_code=404, detail="No project data available. Please submit data first.")
    
    df = pd.DataFrame(project_data_store)
    
    # Need to run full analysis to get metrics
    sentiment_results = df['feedback'].apply(analyze_sentiment_detailed)
    df['sentiment'] = sentiment_results.apply(lambda x: x['sentiment'])
    df['sentiment_score'] = sentiment_results.apply(lambda x: x['confidence'])
    df = calculate_performance_metrics(df)

    student_data = df[df['student'].str.lower() == student_name.lower()]
    
    if student_data.empty:
        raise HTTPException(status_code=404, detail=f"Student '{student_name}' not found")
    
    student_processed = student_data.iloc[0].to_dict()
    
    recommendations = generate_ai_recommendations(student_processed)
    
    return {
        "success": True,
        "student": student_name,
        "data": student_processed,
        "sentiment_analysis": {"sentiment": student_processed['sentiment'], "confidence": student_processed['sentiment_score']},
        "recommendations": recommendations
    }

@app.get("/api/analytics/recommendations/{student_name}")
async def get_recommendations_analytics(student_name: str):
    """Get AI-generated recommendations for a specific student"""
    if not project_data_store:
        raise HTTPException(status_code=404, detail="No project data available")
    
    df = pd.DataFrame(project_data_store)
    sentiment_results = df['feedback'].apply(analyze_sentiment_detailed)
    df['sentiment'] = sentiment_results.apply(lambda x: x['sentiment'])
    df['sentiment_score'] = sentiment_results.apply(lambda x: x['confidence'])

    df = calculate_performance_metrics(df)
    
    student_data = df[df['student'].str.lower() == student_name.lower()]
    
    if student_data.empty:
        raise HTTPException(status_code=404, detail=f"Student '{student_name}' not found")
    
    recommendations = generate_ai_recommendations(student_data.iloc[0].to_dict())
    
    return {
        "success": True,
        "student": student_name,
        "recommendations": recommendations
    }

@app.get("/api/analytics/team-summary")
async def get_team_summary():
    """Get AI-generated team summary"""
    if not project_data_store:
        raise HTTPException(status_code=404, detail="No project data available")
    
    df = pd.DataFrame(project_data_store)
    all_feedback = df['feedback'].tolist()
    team_feedback_summary = summarize_feedback(all_feedback)
    
    sentiment_results = df['feedback'].apply(analyze_sentiment_detailed)
    df['sentiment'] = sentiment_results.apply(lambda x: x['sentiment'])
    df['sentiment_score'] = sentiment_results.apply(lambda x: x['confidence'])
    
    df = calculate_performance_metrics(df)
    
    insights = {
        "team_mood": "Positive" if (df['sentiment'] == 'POSITIVE').sum() > len(df)/2 else "Mixed",
        "average_performance": round(df['quality_score'].mean(), 2),
        "top_performer": df.nlargest(1, 'quality_score')['student'].values[0] if not df.empty else "N/A",
        "most_commits": df.nlargest(1, 'commits')['student'].values[0] if not df.empty else "N/A",
        "feedback_summary": team_feedback_summary,
        "team_health": "Healthy" if df['quality_score'].mean() > 70 else "Needs Attention"
    }
    
    return {
        "success": True,
        "summary": insights,
        "generated_at": datetime.now().isoformat()
    }

@app.get("/api/analytics/risk-assessment")
async def get_risk_assessment():
    """Get comprehensive risk assessment for the team"""
    if not project_data_store:
        raise HTTPException(status_code=404, detail="No project data available")
    
    df = pd.DataFrame(project_data_store)
    
    sentiment_results = df['feedback'].apply(analyze_sentiment_detailed)
    df['sentiment'] = sentiment_results.apply(lambda x: x['sentiment'])
    df['sentiment_score'] = sentiment_results.apply(lambda x: x['confidence'])
    df = calculate_performance_metrics(df)
    
    risk_analysis = []
    for _, row in df.iterrows():
        risk_info = classify_risk_level(row.to_dict())
        risk_analysis.append({
            "student": row['student'],
            "risk_status": risk_info['status'],
            "risk_color": risk_info['color'],
            "ai_classification": risk_info['ai_classification'],
            "confidence": risk_info['confidence']
        })
    
    risk_counts = pd.DataFrame(risk_analysis)['risk_status'].value_counts().to_dict()
    
    return {
        "success": True,
        "individual_risks": risk_analysis,
        "risk_distribution": risk_counts,
        "overall_risk_level": "High" if "🔴 At Risk" in risk_counts and risk_counts["🔴 At Risk"] > 1 else "Low"
    }

@app.post("/api/analytics/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV file with project data"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        required_cols = ['student', 'tasks_assigned', 'tasks_completed', 'commits', 'feedback']
        df.columns = [c.lower().replace('_', '').replace(' ', '') for c in df.columns]

        # Use original names for mapping to Pydantic model
        col_map = {c.lower().replace('_', '').replace(' ', ''): c for c in StudentData.model_fields.keys()}
        
        # Check if required columns are present after normalization
        if not all(col in df.columns for col in required_cols):
             # Try a more forgiving check for common column names
             if not all(c.lower() in [col.lower() for col in df.columns] for c in ['Student', 'Tasks_Assigned', 'Tasks_Completed', 'Commits', 'Feedback']):
                 raise HTTPException(status_code=400, detail=f"CSV must contain columns similar to: {required_cols}")

        students = []
        for _, row in df.iterrows():
            students.append(StudentData(
                student=str(row.get(col_map['student'])),
                tasks_assigned=int(row.get(col_map['tasks_assigned'])),
                tasks_completed=int(row.get(col_map['tasks_completed'])),
                commits=int(row.get(col_map['commits'])),
                feedback=str(row.get(col_map['feedback'])),
                pull_requests=int(row.get(col_map.get('pull_requests', 'pullrequests'), 0)),
                code_reviews=int(row.get(col_map.get('code_reviews', 'codereviews'), 0)),
                bugs_fixed=int(row.get(col_map.get('bugs_fixed', 'bugsfixed'), 0)),
                response_time_hours=float(row.get(col_map.get('response_time_hours', 'responsetimehours'), 0))
            ))
        
        team_data = TeamData(students=students)
        
        result = await analyze_project(team_data)
        
        return {
            "success": True,
            "message": f"CSV uploaded and processed successfully. {len(students)} students analyzed.",
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing error: {str(e)}")

# --- Peer Review Endpoints (from backend_peer.py) ---

@app.post("/api/projects", status_code=201)
def create_project_peer(payload: ProjectCreate):
    """Create a new project"""
    conn = get_conn()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute("INSERT INTO projects (title,description,created_at) VALUES (?,?,?)", (payload.title, payload.description, now))
    pid = c.lastrowid
    conn.commit()
    conn.close()
    return {"status": "ok", "project_id": pid}

@app.post("/api/projects/{project_id}/teammates", status_code=201)
def add_teammate(project_id: int, tm: TeammateIn):
    """Add teammate to a project"""
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id FROM projects WHERE id = ?", (project_id,))
    if not c.fetchone():
        conn.close()
        raise HTTPException(404, "Project not found")
    emb = teammate_embedding_from_skills(tm.skills)
    c.execute(
        "INSERT INTO teammates (project_id, name, role, skills, embedding) VALUES (?,?,?,?,?)",
        (project_id, tm.name, tm.role, json.dumps(tm.skills or {}), ndarray_to_blob(emb))
    )
    tid = c.lastrowid
    conn.commit()
    conn.close()
    return {"status": "ok", "teammate_id": tid}

@app.get("/api/projects")
def list_projects():
    """List all projects with teammates, reviews, and analytics"""
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, title, description, created_at FROM projects ORDER BY created_at DESC")
    projects = []
    for pid, title, desc, created_at in c.fetchall():
        c.execute("SELECT id, name, role, skills FROM teammates WHERE project_id = ?", (pid,))
        trows = c.fetchall()
        teammates = []
        for tid, name, role, skills_json in trows:
            skills = json.loads(skills_json) if skills_json else {}
            teammates.append({"id": tid, "name": name, "role": role, "skills": skills})
        c.execute("SELECT id, reviewer, reviewee, rating, date, comment, skills_json FROM reviews WHERE project_id = ?", (pid,))
        rrows = c.fetchall()
        reviews = []
        for rid, reviewer, reviewee, rating, date, comment, skills_json in rrows:
            skills = json.loads(skills_json) if skills_json else {}
            reviews.append({
                "id": rid,
                "reviewer": reviewer,
                "reviewee": reviewee,
                "rating": rating,
                "date": date,
                "comment": comment,
                "skills": skills
            })
        analytics = project_analytics(pid)
        projects.append({
            "id": pid,
            "title": title,
            "description": desc,
            "created_at": created_at,
            "teammates": teammates,
            "reviews": reviews,
            "analytics": analytics
        })
    conn.close()
    return {"projects": projects}

@app.get("/api/projects/{project_id}")
def get_project(project_id: int):
    """Get details for a single project"""
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, title, description, created_at FROM projects WHERE id = ?", (project_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Project not found")
    pid, title, desc, created_at = row
    c.execute("SELECT id, name, role, skills FROM teammates WHERE project_id = ?", (pid,))
    trows = c.fetchall()
    teammates = []
    for tid, name, role, skills_json in trows:
        skills = json.loads(skills_json) if skills_json else {}
        teammates.append({"id": tid, "name": name, "role": role, "skills": skills})
    c.execute("SELECT id, reviewer, reviewee, rating, date, comment, skills_json FROM reviews WHERE project_id = ?", (pid,))
    rrows = c.fetchall()
    reviews = []
    for rid, reviewer, reviewee, rating, date, comment, skills_json in rrows:
        skills = json.loads(skills_json) if skills_json else {}
        reviews.append({
            "id": rid,
            "reviewer": reviewer,
            "reviewee": reviewee,
            "rating": rating,
            "date": date,
            "comment": comment,
            "skills": skills
        })
    analytics = project_analytics(pid)
    conn.close()
    return {"project": {"id": pid, "title": title, "description": desc, "created_at": created_at, "teammates": teammates, "reviews": reviews, "analytics": analytics}}

@app.post("/api/projects/{project_id}/reviews", status_code=201)
def submit_review(project_id: int, review: ReviewIn):
    """Submit a peer review"""
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id FROM projects WHERE id = ?", (project_id,))
    if not c.fetchone():
        conn.close()
        raise HTTPException(404, "Project not found")
    date_str = review.date or datetime.utcnow().strftime("%Y-%m-%d")
    sent = sentiment_pipe(review.comment or "")
    label = sent[0]["label"]
    score = float(sent[0]["score"])
    c.execute(
        "INSERT INTO reviews (project_id, reviewer, reviewee, rating, date, comment, skills_json, sentiment_label, sentiment_score) VALUES (?,?,?,?,?,?,?,?,?)",
        (project_id, review.reviewer, review.reviewee, float(review.rating), date_str, review.comment, json.dumps(review.skills), label, score)
    )
    rid = c.lastrowid
    conn.commit()
    conn.close()
    return {"status": "ok", "review_id": rid, "sentiment_label": label, "sentiment_score": score}

@app.post("/api/projects/auto_assign")
def auto_assign(req: AutoAssignRequest):
    """Auto-assign reviewers for a project based on project relevance and load"""
    conn = get_conn()
    c = conn.cursor()
    pid = req.project_id
    c.execute("SELECT id FROM projects WHERE id = ?", (pid,))
    if not c.fetchone():
        conn.close()
        raise HTTPException(404, "Project not found")
    
    c.execute("SELECT id, name, skills, embedding FROM teammates WHERE project_id = ?", (pid,))
    trows = c.fetchall()
    if not trows:
        conn.close()
        raise HTTPException(400, "No teammates to assign")
    names, embs = [], []
    for _, name, _, emb_blob in trows:
        names.append(name)
        try:
            embs.append(blob_to_ndarray(emb_blob))
        except Exception:
            embs.append(np.zeros((embedder.get_sentence_embedding_dimension(),)))
    embs = np.vstack(embs)
    
    c.execute("SELECT reviewer_name, COUNT(*) FROM assignments WHERE project_id = ? AND status='assigned' GROUP BY reviewer_name", (pid,))
    loads = {r[0]: r[1] for r in c.fetchall()}
    
    c.execute("SELECT description FROM projects WHERE id = ?", (pid,))
    desc = c.fetchone()[0] or ""
    proj_emb = embedder.encode(desc or "project", normalize_embeddings=True).reshape(1, -1)
    sims = cosine_similarity(proj_emb, embs)[0]
    
    scored = []
    for name, sim in zip(names, sims):
        load = loads.get(name, 0)
        # Composite score: sim - 0.1*load (favor high similarity, penalize high load)
        scored.append((name, sim - 0.1 * load))
    scored_sorted = sorted(scored, key=lambda x: -x[1])
    selected = [s[0] for s in scored_sorted[:req.top_k]]
    assigned_ids = []
    
    # Assumption from original code: auto-assignment is for the entire team/project context.
    # The 'reviewee_name' is blank as it doesn't specify a single reviewee.
    for sel in selected:
        c.execute("INSERT INTO assignments (project_id, reviewer_name, reviewee_name, assigned_at, status) VALUES (?,?,?,?,?)", (pid, sel, "", datetime.utcnow().isoformat(), "assigned"))
        assigned_ids.append(c.lastrowid)
    conn.commit()
    conn.close()
    return {"status": "ok", "assigned_reviewers": selected, "assignment_ids": assigned_ids}

@app.get("/api/assignments")
def list_assignments(project_id: Optional[int] = Query(None)):
    """List all assignments, optionally filtered by project ID"""
    conn = get_conn()
    c = conn.cursor()
    if project_id:
        c.execute("SELECT id, project_id, reviewer_name, reviewee_name, assigned_at, status FROM assignments WHERE project_id = ?", (project_id,))
    else:
        c.execute("SELECT id, project_id, reviewer_name, reviewee_name, assigned_at, status FROM assignments")
    rows = c.fetchall()
    conn.close()
    return {"assignments": [{"id": r[0], "project_id": r[1], "reviewer": r[2], "reviewee": r[3], "assigned_at": r[4], "status": r[5]} for r in rows]}

@app.get("/api/projects/{project_id}/charts/ratings")
def chart_project_ratings(project_id: int):
    """Generate a base64 PNG of the project rating distribution"""
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT rating FROM reviews WHERE project_id = ?", (project_id,))
    rows = c.fetchall()
    conn.close()
    if not rows:
        raise HTTPException(404, "No reviews for project")
    ratings = [r[0] for r in rows]
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(6,3.5))
    ax = sns.histplot(ratings, bins=np.arange(0, 6), discrete=True)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title(f"Project {project_id} Rating Distribution")
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return {"chart_base64": b64}

@app.get("/api/projects/{project_id}/export/reviews")
def export_reviews_csv(project_id: int):
    """Export reviews for a project as a base64 encoded CSV"""
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, reviewer, reviewee, rating, date, comment, skills_json, sentiment_label, sentiment_score FROM reviews WHERE project_id = ?", (project_id,))
    rows = c.fetchall()
    conn.close()
    if not rows:
        raise HTTPException(404, "No reviews to export")
    df = pd.DataFrame(rows, columns=["review_id","reviewer","reviewee","rating","date","comment","skills_json","sentiment_label","sentiment_score"])
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return {"filename": f"project_{project_id}_reviews.csv", "csv_base64": base64.b64encode(csv_bytes).decode("utf-8")}

# --- Health Check (Unified) ---

@app.get("/api/health")
async def health_check():
    """Unified health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": True,
        "recommender_initialized": recommender.df is not None,
        "database_status": "ok",
        "timestamp": datetime.now().isoformat()
    }

# ===========================
# RUN SERVER
# ===========================
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Unified AI Backend API Server...")
    print("📊 All models and services loaded and ready!")
    print("🌐 Server running at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    # Using uvicorn to run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000)