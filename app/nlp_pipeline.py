import logging
import re
from typing import List, Dict, Any
import os
import traceback
import random

# Try to import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("openai package not found. GPT-based insights will not be available.")

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY and OPENAI_AVAILABLE:
    openai.api_key = OPENAI_API_KEY
    logger.info("OpenAI API key is set")
else:
    logger.warning("OPENAI_API_KEY environment variable not set or openai package not available. OpenAI features will not work.")

def extract_insights(transcript_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process transcript segments to extract insights
    This is a simplified version that doesn't depend on transformers
    
    Args:
        transcript_segments: List of transcript segments from YouTube or Whisper
        
    Returns:
        List of dicts with topic info including name, sentiment, and insights
    """
    logger.info("Starting transcript analysis - simplified version")
    
    # Join all segments into a single text
    full_text = " ".join([segment.get("text", "") for segment in transcript_segments])
    
    # Define default topics for demonstration with specific insights for each
    topic_insights = {
        "Climate Change": [
            "Develop a comprehensive carbon footprint tracking system to monitor and reduce greenhouse gas emissions.",
            "Invest in climate resilience measures to prepare facilities for extreme weather events.",
            "Partner with environmental NGOs to support reforestation and carbon offset projects."
        ],
        "Renewable Energy": [
            "Transition facilities to solar or wind power through PPAs (Power Purchase Agreements) with renewable providers.",
            "Install on-site renewable energy generation like solar panels to reduce operational costs.",
            "Implement energy storage solutions to maximize renewable energy utilization during peak demand."
        ],
        "Waste Management": [
            "Establish a zero-waste-to-landfill policy with specific reduction targets and timelines.",
            "Redesign packaging to use biodegradable materials and minimize single-use plastics.",
            "Implement a comprehensive recycling program with clear labeling and employee training."
        ],
        "Carbon Emissions": [
            "Convert company fleet to electric vehicles and install charging infrastructure.",
            "Set science-based targets for emissions reduction aligned with the Paris Agreement.",
            "Implement a carbon pricing mechanism for internal decision-making and investments."
        ],
        "Corporate Sustainability": [
            "Integrate ESG (Environmental, Social, Governance) metrics into executive compensation structures.",
            "Publish an annual sustainability report following GRI or SASB standards for transparency.",
            "Establish a cross-functional sustainability committee with C-suite representation."
        ],
        "Water Conservation": [
            "Install water-efficient fixtures and implement greywater recycling systems in facilities.",
            "Conduct water footprint assessments for all products and set reduction targets.",
            "Partner with watershed conservation organizations in regions where you operate."
        ],
        "Sustainable Supply Chain": [
            "Implement supplier sustainability scorecards and set minimum requirements for vendors.",
            "Reduce transportation emissions through optimized logistics and local sourcing.",
            "Conduct regular sustainability audits of key suppliers and offer improvement resources."
        ],
        "Circular Economy": [
            "Design products for disassembly and recyclability to extend their lifecycle.",
            "Establish product take-back programs to recover materials at end of life.",
            "Launch a service-based business model alongside product sales to reduce resource consumption."
        ]
    }
    
    # Select random subset of topics (5-7) for variety
    selected_topics = random.sample(list(topic_insights.keys()), min(6, len(topic_insights)))
    
    # Generate results with topic-specific insights
    results = []
    for topic in selected_topics:
        try:
            # Try to generate custom insights with OpenAI if available
            if OPENAI_API_KEY and OPENAI_AVAILABLE:
                try:
                    insights = generate_insights_with_gpt(full_text, topic)
                except Exception as e:
                    logger.error(f"Error generating insights with GPT: {e}")
                    # Fall back to predefined insights if GPT fails
                    insights = topic_insights.get(topic, [
                        f"Consider developing sustainability policies related to {topic}.",
                        f"Monitor emerging regulations in the {topic} space to ensure compliance."
                    ])
            else:
                # Use predefined topic-specific insights
                insights = topic_insights.get(topic, [
                    f"Consider developing sustainability policies related to {topic}.",
                    f"Monitor emerging regulations in the {topic} space to ensure compliance."
                ])
            
            # Determine sentiment
            sentiment = analyze_sentiment_simple(full_text, topic)
            
            # Add to results
            results.append({
                "name": topic,
                "sentiment": sentiment,
                "insights": insights
            })
            
        except Exception as e:
            logger.error(f"Error processing topic '{topic}': {str(e)}")
            # Add a fallback result if a topic fails
            results.append({
                "name": topic,
                "sentiment": "neutral",
                "insights": [f"Consider developing sustainability policies related to {topic} based on emerging regulations."]
            })
    
    logger.info(f"Completed transcript analysis, found {len(results)} topics")
    return results

def analyze_sentiment_simple(text: str, topic: str) -> str:
    """
    Simple sentiment analysis using keyword matching
    
    Args:
        text: Text to analyze
        topic: Topic to consider for context
        
    Returns:
        Sentiment label ("positive", "negative", or "neutral")
    """
    logger.info(f"Using simple keyword-based sentiment analysis for topic: {topic}")
    
    # Lists of positive and negative sentiment words
    positive_words = [
        "benefit", "advantage", "opportunity", "positive", "good", "great", "excellent", 
        "improve", "improvement", "progress", "solution", "success", "successful", 
        "effective", "efficient", "sustainable", "innovative", "innovation", "clean",
        "save", "savings", "growth", "advance", "advantage", "beneficial", "better",
        "progress", "promising", "protect", "protection", "reduce", "reduction"
    ]
    
    negative_words = [
        "problem", "challenge", "difficult", "difficulty", "issue", "concern", "risk",
        "threat", "crisis", "damage", "harmful", "negative", "pollute", "pollution",
        "waste", "danger", "dangerous", "hazard", "hazardous", "toxic", "costly",
        "expensive", "fail", "failure", "loss", "deficit", "shortage", "decrease"
    ]
    
    # Normalize text
    text_lower = text.lower()
    
    # Count occurrences of sentiment words
    positive_count = sum(text_lower.count(word) for word in positive_words)
    negative_count = sum(text_lower.count(word) for word in negative_words)
    
    # Determine sentiment
    if positive_count > negative_count * 1.5:  # Significantly more positive
        return "positive"
    elif negative_count > positive_count * 1.5:  # Significantly more negative
        return "negative"
    else:
        return "neutral"  # Balanced or insufficient sentiment words

def generate_insights_with_gpt(text: str, topic: str) -> List[str]:
    """
    Generate insights for a topic using GPT-4 or similar model
    
    Args:
        text: Transcript text
        topic: Topic to generate insights for
        
    Returns:
        List of insight statements
    """
    try:
        logger.info(f"Generating insights for topic: {topic}")
        
        # Check if OpenAI API is available
        if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
            logger.warning("OpenAI API not available. Returning default insights.")
            return [
                f"Consider developing comprehensive sustainability policies related to {topic}.",
                f"Monitor emerging regulations in the {topic} space to ensure compliance.",
                f"Engage with industry partners on collaborative {topic} initiatives."
            ]
        
        # Limit text length to avoid token limits
        if len(text) > 6000:
            text = text[:6000] + "..."
        
        # Different prompt templates for different topics to ensure variety
        topic_prompts = {
            "Climate Change": f"""
            Based on this meeting transcript excerpt about sustainability:
            
            {text}
            
            Generate 3 specific and actionable climate change mitigation strategies that businesses could implement. 
            Focus on emissions reduction, adaptation measures, and policy alignment. 
            Format each recommendation as a clear, concise bullet point.
            """,
            
            "Renewable Energy": f"""
            After analyzing this sustainability meeting transcript:
            
            {text}
            
            Provide 3 concrete recommendations for businesses to transition to renewable energy.
            Include considerations for implementation costs, ROI timeframes, and regulatory incentives.
            Format each recommendation as a concise bullet point.
            """,
            
            "Waste Management": f"""
            Based on this sustainability discussion:
            
            {text}
            
            Outline 3 innovative waste reduction and circular economy strategies for businesses.
            Consider waste stream analysis, material recapture, and supply chain optimization.
            Present each strategy as a clear action item with expected outcomes.
            """,
            
            "Carbon Emissions": f"""
            From this transcript of a sustainability meeting:
            
            {text}
            
            Detail 3 practical carbon reduction initiatives that businesses could implement within the next 12 months.
            Include measurement methodologies, reduction technologies, and reporting frameworks.
            Format as actionable bullet points with implementation considerations.
            """,
            
            "Corporate Sustainability": f"""
            From the attached sustainability discussion transcript:
            
            {text}
            
            Recommend 3 strategic approaches to embed sustainability into corporate governance and operations.
            Address stakeholder engagement, ESG metrics integration, and transparency practices.
            Format each recommendation as a concrete action item with expected benefits.
            """,
            
            "Water Conservation": f"""
            Based on this sustainability meeting:
            
            {text}
            
            Propose 3 water stewardship initiatives that businesses could implement to reduce consumption and pollution.
            Focus on measurement, reduction technologies, and watershed protection.
            Format as specific, actionable bullet points.
            """,
            
            "Sustainable Supply Chain": f"""
            From this sustainability discussion transcript:
            
            {text}
            
            Outline 3 practical approaches to improve supply chain sustainability and resilience.
            Consider supplier engagement, transparency requirements, and material sourcing practices.
            Format as concrete action steps with implementation guidelines.
            """,
            
            "Circular Economy": f"""
            Based on this meeting transcript about sustainability:
            
            {text}
            
            Recommend 3 innovative circular economy business models or practices that companies could adopt.
            Include product design considerations, material recovery systems, and service-based alternatives.
            Format as specific strategic initiatives with expected sustainability impacts.
            """
        }
        
        # Select appropriate prompt or use default
        prompt = topic_prompts.get(topic, f"""
        Topic: {topic}
        
        Based on this excerpt from a public meeting transcript on sustainability:
        
        {text}
        
        Generate 3 specific, actionable recommendations related to {topic} that businesses can implement.
        Each recommendation should be concrete, feasible, and directly related to sustainability.
        Format each as a concise bullet point starting with an action verb.
        """)
        
        # Call GPT API
        logger.info("Calling OpenAI API for insight generation")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert sustainability consultant who extracts actionable insights from public meeting transcripts. Your recommendations are specific, diverse, and tailored to each sustainability topic."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # Extract insights from response
        insights_text = response.choices[0].message.content.strip()
        
        # Parse bullet points
        insights = [line.strip().replace('• ', '').replace('- ', '') 
                   for line in insights_text.split('\n') 
                   if line.strip().startswith(('•', '-'))]
        
        # If no bullet points detected, split by newlines and take non-empty lines
        if not insights:
            insights = [line.strip() for line in insights_text.split('\n') if line.strip()]
        
        # Ensure we have at least one insight
        if not insights:
            insights = [insights_text]
            
        logger.info(f"Generated {len(insights)} insights for topic '{topic}'")
        return insights
    
    except Exception as e:
        logger.error(f"Error generating insights with GPT for topic '{topic}': {str(e)}\n{traceback.format_exc()}")
        # Fallback if GPT fails
        return [
            f"Consider developing comprehensive sustainability policies related to {topic}.",
            f"Monitor emerging regulations in the {topic} space to ensure compliance."
        ] 