import google.generativeai as genai
from .config import GEMINI_API_KEY, GEMINI_MODEL_NAME
from app.logger import logger

# Configure Gemini API with key from config
genai.configure(api_key=GEMINI_API_KEY)

class LLM:
    """Handles interaction with Google's Gemini LLM for report generation."""
    
    def __init__(self):
        """Initialize the LLM model."""
        try:
            self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            logger.info("Gemini LLM model initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize LLM model", exc_info=True)
            raise Exception("Error initializing LLM model. Check API key and model name.")

    def prompt_template(self, result, language="English"):
        """
        Generate a structured prompt for the LLM.
        Expected output should be around 600 words with detailed analysis.
        """
        prompt = f"""You are a cardiologist specializing in heart disease risk assessment and prevention. Based on the provided data, generate a comprehensive report (approximately 600 words) covering these sections:

1. Prediction Summary (50-75 words):
   - State the heart disease prediction model result clearly
   - Confidence level assessment
   - Initial risk evaluation based on clinical features

2. Clinical Analysis (100-125 words):
   - Detailed analysis of key risk factors present
   - Cardiovascular health indicators
   - Interpretation of vital signs and test results
   - Correlation between different risk factors
   - Severity assessment of the condition

3. Risk Factor Assessment (100-125 words):
   - Analysis of modifiable risk factors (lifestyle, diet, etc.)
   - Analysis of non-modifiable risk factors (age, gender, family history)
   - Comparative risk analysis with general population
   - Short-term vs. long-term risk projections
   - Potential comorbidity considerations

4. Differential Considerations (100-125 words):
   - Other possible cardiovascular conditions to consider
   - Common conditions with similar presentations
   - Age and gender-specific considerations
   - Stress-related cardiac conditions
   - Potential non-cardiac causes of symptoms

5. Management Recommendations (150-175 words):
   A. Immediate Actions:
      - Monitoring parameters
      - Essential lifestyle modifications
      - Medication considerations (general classes, not specific prescriptions)
   
   B. Preventive Measures:
      - Diet and nutrition guidelines
      - Exercise recommendations
      - Stress management strategies
      - Sleep hygiene improvements
   
   C. Long-term Management:
      - Regular screening schedule
      - Health metrics to track
      - Specialist consultation recommendations

6. Follow-up Protocol (50-75 words):
   - Monitoring timeline
   - Warning signs requiring immediate medical attention
   - Recommended follow-up tests
   - Documentation and health tracking suggestions

Input Data:
{result}

IMPORTANT GUIDELINES:
1. Provide the report in {language} language
2. Maintain all six sections with their exact headings and don't mention no.of words in the headings or in the report
3. Include specific numbers, measurements, and timelines where applicable
4. Consider both individual health factors and population-level statistics
5. Provide actionable, practical recommendations
6. Use bullet points for clarity where appropriate
7. Include evidence-based recommendations
8. Consider accessibility of healthcare resources
9. Address both immediate and long-term management strategies
10. Do not add any additional sections or explanatory text like "Here is the report in {language} language" or anything like that.
11. Provide the report in markdown format
12. Do not provide specific medication prescriptions or dosages
13. Use consistent heading levels throughout the report:
    - For Diagnostic Report heading, use level 1 heading (#)
    - Main section titles (1-6) should be level 2 headings (##)
    - Subsection titles (A, B, C under Management Recommendations) should be level 3 headings (###)
    - Do not use level 1 headings (#) anywhere in the report except for Diagnostic Report heading
    - Do not use any heading levels beyond level 3 (###)
14. Ensure all headings follow this exact format with no variations

Format each section clearly with headers and subheaders for easy reading. Use bullet points for lists and recommendations. Highlight critical information using bold text (**important text**)."""

        return prompt

    def inference(self, result: str, language: str) -> str:
        """
        Generate a report using the LLM model.
        
        Args:
            result: Analysis results and clinical data
            language: Target language for the report
        """
        try:
            # Generate prompt using template
            refined_prompt = self.prompt_template(result, language)
    
            prompt = [{'role': 'user', 'parts': [refined_prompt]}]
            
            # Generate response
            logger.info("Sending request to Gemini LLM")
            response = self.model.generate_content(prompt)
            
            if response.text:
                llm_response = response.text
                llm_response = llm_response.replace("```markdown", "").replace("```", "")
                logger.info("Successfully generated LLM report")
                return llm_response
            else:
                logger.error("LLM returned empty response")
                raise Exception("LLM response is empty")

        except Exception as e:
            logger.error("Error during LLM inference", exc_info=True)
            raise Exception(f"Error during LLM inference: {str(e)}")
