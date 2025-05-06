from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import traceback
import json
import redis
import sys
import os

# Add the current directory to the path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from youtube_client import extract_video_id, fetch_transcript
from nlp_pipeline import extract_insights

# Configure logging to show more details
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Redis client - adjust connection details as needed
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Define request model
class VideoRequest(BaseModel):
    youtube_url: str

# Initialize FastAPI app
app = FastAPI(title="Public Meeting Insights API")

# Configure CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "online", "message": "Public Meeting Insights API is running"}

@app.post("/videos/process")
async def process_video_request(video_url: str, background_tasks: BackgroundTasks):
    """Process a video URL in the background"""
    try:
        # Extract video ID from URL
        video_id = extract_video_id(video_url)
        
        # Clear any stale keys
        redis_client.delete(f"status:{video_id}")
        redis_client.delete(f"error:{video_id}")
        logger.info(f"Endpoint cleared Redis keys for video_id: {video_id}")
        
        # Set initial status
        redis_client.setex(f"status:{video_id}", 3600, "pending")
        logger.info(f"Endpoint set status:{video_id}=pending (TTL: 3600s)")
        
        # Start background task
        background_tasks.add_task(process_video, video_id, video_url)
        
        return {"video_id": video_id, "status": "pending"}
        
    except Exception as e:
        logger.error(f"Error processing video URL {video_url}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/videos/status/{video_id}")
async def get_video_status(video_id: str):
    """Get the processing status of a video"""
    status = redis_client.get(f"status:{video_id}")
    if not status:
        return {"status": "not_found"}
    
    return {"status": status}

@app.get("/videos/insights/{video_id}")
async def get_video_insights(video_id: str):
    """Get the insights for a processed video"""
    status = redis_client.get(f"status:{video_id}")
    if not status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if status != "completed":
        raise HTTPException(status_code=400, detail=f"Video processing is {status}")
    
    insights = redis_client.get(f"insights:{video_id}")
    if not insights:
        raise HTTPException(status_code=404, detail="Insights not found")
    
    return json.loads(insights)

@app.get("/videos/error/{video_id}")
async def get_video_error(video_id: str):
    """Get error information for a failed video processing"""
    status = redis_client.get(f"status:{video_id}")
    if not status or status != "error":
        raise HTTPException(status_code=404, detail="No error found")
    
    error = redis_client.get(f"error:{video_id}")
    return {"error": error or "Unknown error"}

@app.post("/process")
async def process_video_endpoint(request: VideoRequest):
    """
    Process a YouTube video to extract insights
    
    Args:
        request: VideoRequest object containing youtube_url
        
    Returns:
        JSON with topics, sentiments, insights and full transcript
    """
    try:
        logger.info(f"Processing YouTube URL: {request.youtube_url}")
        
        # Extract video ID
        video_id = extract_video_id(request.youtube_url)
        
        # Clear any stale keys
        redis_client.delete(f"status:{video_id}")
        redis_client.delete(f"error:{video_id}")
        logger.info(f"Process endpoint cleared Redis keys for video_id: {video_id}")
        
        # Fetch transcript from YouTube
        logger.info("Attempting to fetch transcript...")
        transcript_data = fetch_transcript(request.youtube_url, redis_client)
        
        # Handle both dictionary and list response formats
        transcript_segments = []
        if isinstance(transcript_data, dict) and "segments" in transcript_data:
            transcript_segments = transcript_data["segments"]
            transcript_source = transcript_data.get("source", "unknown")
        elif isinstance(transcript_data, list):
            # If transcript_data is already a list of segments
            transcript_segments = transcript_data
            transcript_source = "direct"
        else:
            raise ValueError("Failed to retrieve transcript - invalid format")
            
        if not transcript_segments:
            raise ValueError("Failed to retrieve transcript - empty segments")
        
        # Extract full transcript text for response
        full_transcript = " ".join([segment["text"] for segment in transcript_segments])
        logger.info(f"Successfully fetched transcript from {transcript_source}. Length: {len(full_transcript)} characters")
        
        # Process transcript to extract insights
        logger.info("Starting to extract insights from transcript...")
        analysis_results = extract_insights(transcript_segments)
        logger.info(f"Successfully extracted insights. Found {len(analysis_results)} topics")
        
        # Return combined results
        return {
            "topics": analysis_results,
            "transcript": full_transcript
        }
    except Exception as e:
        error_detail = f"Error processing video: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

async def process_video(video_id: str, video_url: str):
    """Background task to process a video"""
    try:
        # Clear any stale keys at the start of processing
        redis_client.delete(f"status:{video_id}")
        redis_client.delete(f"error:{video_id}")
        logger.info(f"Task cleared Redis keys for video_id: {video_id}")
        
        # Set processing status
        redis_client.setex(f"status:{video_id}", 3600, "processing")
        logger.info(f"Task set status:{video_id}=processing (TTL: 3600s)")
        
        # Get transcript
        transcript_data = fetch_transcript(video_url, redis_client)
        transcript_segments = transcript_data["segments"]
        
        # Extract insights
        insights = extract_insights(transcript_segments)
        
        # Cache insights
        redis_client.setex(f"insights:{video_id}", 86400, json.dumps(insights))
        logger.info(f"Cached insights for video_id: {video_id} (TTL: 86400s)")
        
        # Update status to completed
        redis_client.setex(f"status:{video_id}", 3600, "completed")
        logger.info(f"Task set status:{video_id}=completed (TTL: 3600s)")
        
    except Exception as e:
        logger.error(f"Error in background processing for video_id {video_id}: {str(e)}")
        
        # Set error status
        redis_client.setex(f"error:{video_id}", 3600, str(e))
        redis_client.setex(f"status:{video_id}", 3600, "error")
        logger.info(f"Task set error:{video_id}='{str(e)}' and status:{video_id}=error (TTL: 3600s)")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 