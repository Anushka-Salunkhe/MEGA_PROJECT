"""
TruthLens AI — FastAPI High-Performance Backend
===============================================
REST API Endpoints:
  POST /api/verify/text
  POST /api/verify/image
  POST /api/verify/url
  POST /api/verify/audio
  POST /api/verify/video
  POST /api/verify/multimodal
  GET  /api/history
  GET  /api/history/{id}
  GET  /api/evaluation
  POST /api/evaluation/run
  GET  /api/dashboard/stats
  GET  /api/health
"""

import os
import sys
import shutil
import uuid
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Depends, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from storage.database import init_db, SessionLocal, VerificationRecord, BenchmarkEvaluationRecord
from services.verification_service import VerificationService
from services.evaluation_service import EvaluationService

# Ensure storage and upload directories exist
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize database schema
init_db()

app = FastAPI(
    title="TruthLens AI API",
    description="Multimodal Misinformation Verification System Backend",
    version="2.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static uploads directory for browser previews
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

# Services
verification_service = VerificationService()
evaluation_service = EvaluationService()

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB



def save_upload(file: UploadFile) -> Tuple[str, str, int, str]:
    """Helper to safely persist uploaded files."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Empty or invalid file upload.")

    ext = os.path.splitext(file.filename)[1].lower()
    unique_filename = f"{uuid.uuid4().hex[:10]}_{file.filename}"
    file_path = os.path.join(UPLOADS_DIR, unique_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE_BYTES:
        os.remove(file_path)
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed upload size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."
        )

    return file_path, file.filename, file_size, file.content_type or ext


# ----------------------------------------------------
# Pydantic Schemas
# ----------------------------------------------------
class TextVerificationRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Article text or headline to analyze.")
    engine_version: Optional[str] = "v2"

class URLVerificationRequest(BaseModel):
    url: str = Field(..., min_length=5, description="Article or claim URL.")
    text: Optional[str] = None
    engine_version: Optional[str] = "v2"

class EvaluationRunRequest(BaseModel):
    dataset_key: str = Field(default="test", description="Dataset key: test, politifact, gossipcop, cleaned_text")
    sample_limit: Optional[int] = Field(default=None, description="Optional sample limit for faster evaluation")

class BenchmarkRunRequest(BaseModel):
    benchmark_key: str = Field(default="multimodal", description="Benchmark key: text_only, image_only, video_only, audio_only, url_only, text_image, text_url, image_url, text_image_url, multimodal, or all")
    dataset_key: str = Field(default="test", description="Dataset key: test, politifact, gossipcop, cleaned_text")
    sample_limit: Optional[int] = Field(default=None, description="Optional sample limit for faster evaluation")


# ----------------------------------------------------
# Health & Status Endpoint
# ----------------------------------------------------
@app.get("/api/health")
def health_check():
    """System health check and operational status."""
    # Probe live component status
    components: Dict[str, Any] = {}
    all_ok = True

    # --- text pipeline ---
    try:
        te = getattr(verification_service, "text_encoder_v2", None)
        components["text_encoder"] = {
            "status": "ok" if te is not None else "degraded",
            "model": "RoBERTa-base (768-dim)",
        }
    except Exception as exc:
        components["text_encoder"] = {"status": "error", "detail": str(exc)}
        all_ok = False

    # --- image encoder ---
    try:
        ie = getattr(verification_service, "image_encoder_v2", None)
        components["image_encoder"] = {
            "status": "ok" if ie is not None else "degraded",
            "model": "ViT-base-patch16-224 (768-dim)",
        }
    except Exception as exc:
        components["image_encoder"] = {"status": "error", "detail": str(exc)}
        all_ok = False

    # --- audio processor ---
    try:
        from utils.audio_processor import processor as ap
        components["audio_processor"] = {
            "status": "ok" if ap is not None else "degraded",
            "model": "Whisper-tiny (Speech-to-Text)",
        }
    except Exception as exc:
        components["audio_processor"] = {"status": "error", "detail": str(exc)}
        all_ok = False

    # --- fusion engine ---
    try:
        fe = getattr(verification_service, "fusion_engine", None)
        components["fusion_engine"] = {
            "status": "ok" if fe is not None else "degraded",
            "type": type(fe).__name__ if fe else "AdaptiveMultimodalFusion",
        }
    except Exception as exc:
        components["fusion_engine"] = {"status": "error", "detail": str(exc)}
        all_ok = False

    # --- media forensics ---
    try:
        mf = getattr(verification_service, "media_forensics", None)
        components["media_forensics"] = {"status": "ok" if mf is not None else "degraded"}
    except Exception as exc:
        components["media_forensics"] = {"status": "error", "detail": str(exc)}
        all_ok = False

    # --- decision engine ---
    try:
        de = getattr(verification_service, "decision_engine", None)
        components["decision_engine"] = {
            "status": "ok" if de is not None else "degraded",
            "type": type(de).__name__ if de else "UnifiedDecisionEngine",
            "verdict_labels": ["REAL", "FAKE", "MISLEADING", "UNVERIFIED", "INSUFFICIENT EVIDENCE"],
        }
    except Exception as exc:
        components["decision_engine"] = {"status": "error", "detail": str(exc)}
        all_ok = False

    # --- database ---
    try:
        db = SessionLocal()
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db.close()
        components["database"] = {"status": "ok", "backend": "SQLite"}
    except Exception as exc:
        components["database"] = {"status": "error", "detail": str(exc)}
        all_ok = False

    return {
        "status": "healthy" if all_ok else "degraded",
        "service": "TruthLens AI",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "all_systems_ready": all_ok,
        "models": {
            "text": "RoBERTa-base (768-dim)",
            "image": "ViT-base-patch16-224 (768-dim)",
            "audio": "Whisper-tiny (Speech-to-Text)",
            "evidence_nli": "cross-encoder/nli-roberta-base",
            "fusion": "Adaptive Reliability-Aware Fusion + XGBoost",
            "explainability": "SHAP TreeExplainer",
        },
        "components": components,
        "capabilities": {
            "text_verification": True,
            "image_verification": True,
            "url_intelligence": True,
            "audio_transcription": True,
            "video_temporal_sampling": True,
            "missing_modality_robustness": True,
            "verdict_labels": ["REAL", "FAKE", "MISLEADING", "UNVERIFIED", "INSUFFICIENT EVIDENCE"],
        },
    }


# ----------------------------------------------------
# 1. TEXT VERIFICATION
# ----------------------------------------------------
@app.post("/api/verify/text")
def verify_text(req: TextVerificationRequest):
    """Verify claim or article text."""
    try:
        result = verification_service.verify(
            text=req.text,
            engine_version=req.engine_version or "v2"
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------
# 2. IMAGE VERIFICATION
# ----------------------------------------------------
@app.post("/api/verify/image")
def verify_image(
    image: UploadFile = File(...),
    text: Optional[str] = Form(None),
    engine_version: str = Form("v2")
):
    """Verify uploaded image and optional accompanying caption/claim text."""
    allowed_exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    ext = os.path.splitext(image.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {ext}. Allowed: {allowed_exts}"
        )

    file_path, fname, fsize, ftype = save_upload(image)
    try:
        result = verification_service.verify(
            text=text,
            image_path=file_path,
            engine_version=engine_version,
            file_metadata={"name": fname, "size": fsize, "type": ftype}
        )
        # Add relative URL for web display
        result["image_display_url"] = f"/uploads/{os.path.basename(file_path)}"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------
# 3. URL VERIFICATION
# ----------------------------------------------------
@app.post("/api/verify/url")
def verify_url(req: URLVerificationRequest):
    """Verify news source URL and domain reliability."""
    try:
        result = verification_service.verify(
            text=req.text,
            url=req.url,
            engine_version=req.engine_version or "v2"
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------
# 4. AUDIO VERIFICATION
# ----------------------------------------------------
@app.post("/api/verify/audio")
def verify_audio(
    audio: UploadFile = File(...),
    text: Optional[str] = Form(None),
    engine_version: str = Form("v2")
):
    """Verify speech audio recording via Whisper-tiny transcription and acoustic reliability."""
    allowed_exts = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
    ext = os.path.splitext(audio.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {ext}. Allowed: {allowed_exts}"
        )

    file_path, fname, fsize, ftype = save_upload(audio)
    try:
        result = verification_service.verify(
            text=text,
            audio_path=file_path,
            engine_version=engine_version,
            file_metadata={"name": fname, "size": fsize, "type": ftype}
        )
        result["audio_display_url"] = f"/uploads/{os.path.basename(file_path)}"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------
# 5. VIDEO VERIFICATION
# ----------------------------------------------------
@app.post("/api/verify/video")
def verify_video(
    video: UploadFile = File(...),
    text: Optional[str] = Form(None),
    engine_version: str = Form("v2")
):
    """Verify news video recording with uniform frame sampling and visual consistency."""
    allowed_exts = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    ext = os.path.splitext(video.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {ext}. Allowed: {allowed_exts}"
        )

    file_path, fname, fsize, ftype = save_upload(video)
    try:
        result = verification_service.verify(
            text=text,
            video_path=file_path,
            engine_version=engine_version,
            file_metadata={"name": fname, "size": fsize, "type": ftype}
        )
        result["video_display_url"] = f"/uploads/{os.path.basename(file_path)}"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------
# 6. UNIFIED MULTIMODAL VERIFICATION
# ----------------------------------------------------
@app.post("/api/verify/multimodal")
def verify_multimodal(
    text: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    engine_version: str = Form("v2")
):
    """Unified endpoint accepting any combination of Text, Image, URL, Audio, and Video."""
    img_path = None
    aud_path = None
    vid_path = None
    fmeta = {}

    if image and image.filename:
        img_path, fn, fs, ft = save_upload(image)
        fmeta["image"] = {"name": fn, "size": fs, "type": ft}

    if audio and audio.filename:
        aud_path, fn, fs, ft = save_upload(audio)
        fmeta["audio"] = {"name": fn, "size": fs, "type": ft}

    if video and video.filename:
        vid_path, fn, fs, ft = save_upload(video)
        fmeta["video"] = {"name": fn, "size": fs, "type": ft}

    try:
        result = verification_service.verify(
            text=text,
            image_path=img_path,
            url=url,
            audio_path=aud_path,
            video_path=vid_path,
            engine_version=engine_version,
            file_metadata=fmeta.get("image") or fmeta.get("video") or fmeta.get("audio")
        )
        if img_path:
            result["image_display_url"] = f"/uploads/{os.path.basename(img_path)}"
        if aud_path:
            result["audio_display_url"] = f"/uploads/{os.path.basename(aud_path)}"
        if vid_path:
            result["video_display_url"] = f"/uploads/{os.path.basename(vid_path)}"

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------
# 7. HISTORY ENDPOINTS
# ----------------------------------------------------
@app.get("/api/history")
def get_history(
    verdict: Optional[str] = Query(None, description="Filter by verdict: REAL, FAKE, UNCERTAIN"),
    search: Optional[str] = Query(None, description="Search claim text or URL"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0)
):
    """Retrieve historical verification records with search and verdict filtering."""
    db = SessionLocal()
    try:
        query = db.query(VerificationRecord).order_by(VerificationRecord.timestamp.desc())

        if verdict and verdict.upper() in ["REAL", "FAKE", "UNCERTAIN"]:
            query = query.filter(VerificationRecord.prediction == verdict.upper())

        if search and search.strip():
            term = f"%{search.strip()}%"
            query = query.filter(
                (VerificationRecord.claim_text.ilike(term)) |
                (VerificationRecord.news_url.ilike(term)) |
                (VerificationRecord.verification_id.ilike(term))
            )

        total = query.count()
        records = query.offset(offset).limit(limit).all()

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": [r.to_dict() for r in records]
        }
    finally:
        db.close()


@app.get("/api/history/{id}")
def get_history_by_id(id: str):
    """Retrieve full verification audit details by verification_id or primary key id."""
    db = SessionLocal()
    try:
        if id.isdigit():
            rec = db.query(VerificationRecord).filter(VerificationRecord.id == int(id)).first()
        else:
            rec = db.query(VerificationRecord).filter(VerificationRecord.verification_id == id).first()

        if not rec:
            raise HTTPException(status_code=404, detail="Verification record not found.")

        return rec.to_dict()
    finally:
        db.close()


# ----------------------------------------------------
# 8. DASHBOARD OVERVIEW ENDPOINT
# ----------------------------------------------------
@app.get("/api/dashboard/stats")
def get_dashboard_stats():
    """Retrieve overview statistics computed from actual database records."""
    db = SessionLocal()
    try:
        stats = evaluation_service.get_runtime_statistics()
        # Recent 5 verifications
        recent = db.query(VerificationRecord).order_by(VerificationRecord.timestamp.desc()).limit(5).all()
        stats["recent_verifications"] = [r.to_dict() for r in recent]
        return stats
    finally:
        db.close()


# ----------------------------------------------------
# 9. EVALUATION & BENCHMARKING ENDPOINTS
# ----------------------------------------------------
@app.get("/api/evaluation")
def get_evaluation_overview(dataset: Optional[str] = Query(None)):
    """
    Get evaluation page information distinguishing:
      A. Runtime Verification Statistics
      B. Benchmark Evaluation Results
    """
    runtime_stats = evaluation_service.get_runtime_statistics()
    benchmark_res = evaluation_service.get_latest_benchmark_evaluation(dataset)

    return {
        "runtime_statistics": runtime_stats,
        "benchmark_evaluation": benchmark_res,
        "available_datasets": evaluation_service.AVAILABLE_DATASETS,
        "audio_video_benchmark_notice": "Ground-truth labels unavailable for audio/video collections — benchmark accuracy not calculated."
    }


@app.post("/api/evaluation/run")
def run_evaluation(req: EvaluationRunRequest):
    """
    Run an actual evaluation against the selected labeled dataset.
    Never fabricates metrics.
    """
    try:
        result = evaluation_service.run_evaluation(
            dataset_key=req.dataset_key,
            sample_limit=req.sample_limit
        )
        return {
            "status": "success",
            "message": f"Successfully evaluated dataset '{req.dataset_key}'.",
            "evaluation_result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------
# 10. DEDICATED MODALITY BENCHMARKING ENDPOINTS
# ----------------------------------------------------
@app.get("/api/benchmarks")
def list_benchmarks():
    """
    List all 10 supported modality benchmarks with operational readiness,
    modality pipelines, and descriptions.
    """
    return {
        "status": "success",
        "benchmarks": evaluation_service.list_benchmarks(),
        "available_datasets": evaluation_service.AVAILABLE_DATASETS
    }


@app.post("/api/benchmarks/run")
def run_benchmark_endpoint(req: BenchmarkRunRequest):
    """
    Run evaluation for an individual modality benchmark or 'all' available benchmarks.
    Never hardcodes metrics; evaluates on ground-truth labeled datasets.
    """
    try:
        if req.benchmark_key.lower() == "all":
            result = evaluation_service.run_all_benchmarks(
                dataset_key=req.dataset_key,
                sample_limit=req.sample_limit
            )
            return result
        else:
            result = evaluation_service.run_single_benchmark(
                benchmark_key=req.benchmark_key,
                dataset_key=req.dataset_key,
                sample_limit=req.sample_limit
            )
            return {
                "status": "success",
                "message": f"Successfully evaluated benchmark '{req.benchmark_key}'.",
                "result": result
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/benchmarks/results")
def get_benchmark_results(
    dataset: str = Query("test", description="Dataset key: test, politifact, gossipcop, cleaned_text"),
    benchmark_key: Optional[str] = Query(None, description="Optional benchmark key filter")
):
    """
    Retrieve the latest evaluated result for a specific benchmark or dataset.
    """
    try:
        result = evaluation_service.get_latest_benchmark_evaluation(
            dataset_name=dataset,
            benchmark_key=benchmark_key
        )
        if not result:
            raise HTTPException(status_code=404, detail="No evaluation results found for specified parameters.")
        return {
            "status": "success",
            "result": result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/benchmarks/compare")
def get_benchmark_comparison(
    dataset: str = Query("test", description="Dataset key: test, politifact, gossipcop, cleaned_text")
):
    """
    Retrieve comparative performance data across all 10 benchmarks.
    Formatted specifically for Graphs A through E (Accuracy, Precision, Recall, F1, ROC-AUC)
    and the comprehensive comparison table.
    """
    try:
        comparison = evaluation_service.get_benchmark_comparison(dataset_key=dataset)
        return {
            "status": "success",
            "comparison": comparison
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/benchmarks/{benchmark_key}/curves")
def get_benchmark_curves(
    benchmark_key: str,
    dataset: str = Query("test", description="Dataset key")
):
    """
    Retrieve ROC curve, Precision-Recall curve, and Calibration diagram points for a benchmark.
    """
    try:
        curves = evaluation_service.get_benchmark_curves(benchmark_key=benchmark_key, dataset_key=dataset)
        return {
            "status": "success",
            "benchmark_key": benchmark_key,
            "curves": curves
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/benchmarks/history")
def get_benchmark_history(
    limit: int = Query(20, ge=1, le=100)
):
    """
    Retrieve historical benchmark execution audit records from the database.
    """
    db = SessionLocal()
    try:
        records = (
            db.query(BenchmarkEvaluationRecord)
            .order_by(BenchmarkEvaluationRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        return {
            "status": "success",
            "total": len(records),
            "history": [r.to_dict() for r in records]
        }
    finally:
        db.close()


@app.get("/api/benchmarks/export")
def export_benchmarks(
    dataset: str = Query("test", description="Dataset key"),
    format: str = Query("json", description="Export format: json or csv")
):
    """
    Export benchmark evaluation results in CSV or JSON format.
    """
    try:
        content = evaluation_service.export_benchmark_results(
            dataset_key=dataset,
            export_format=format
        )
        media_type = "text/csv" if format.lower() == "csv" else "application/json"
        filename = f"truthlens_benchmarks_{dataset}.{format.lower()}"
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[PROJECT_ROOT, os.path.join(PROJECT_ROOT, "services"), os.path.join(PROJECT_ROOT, "modules")]
    )
