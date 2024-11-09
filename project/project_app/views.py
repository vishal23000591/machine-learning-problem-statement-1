from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import google.generativeai as genai
import csv
import re
import os
from PIL import Image
import tempfile
from .models import MedicalAnalysis
from paddleocr import PaddleOCR
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the absolute path of the current file's directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class MedicalKeywordAnalyzer:
    def __init__(self, csv_path):
        # Construct the absolute path to the CSV file
        self.csv_path = os.path.join(CURRENT_DIR, 'data', 'medical_keywords_dataset.csv')
        print(f"Looking for CSV file at: {self.csv_path}")  # Debug print
        self.medical_keywords = self.load_medical_keywords(csv_path)
        
    def load_medical_keywords(self, filename):
        medical_keywords = {}
        with open(filename, newline='') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                category, term = row
                if category not in medical_keywords:
                    medical_keywords[category] = set()
                medical_keywords[category].add(term.lower())
        return medical_keywords
    
    def generate_boolean_query(self, user_query):
        terms_found = []
        user_query = user_query.lower()
        
        for terms in self.medical_keywords.values():
            for term in terms:
                if re.search(r'\b' + re.escape(term) + r'\b', user_query):
                    terms_found.append(term)
        
        return " OR ".join(terms_found) if terms_found else "No relevant terms found."

class MedicalImageAnalyzer:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        
    def analyze_health_data(self, image_path, disease_type):
        result = self.ocr.ocr(image_path, cls=True)
        text_data = " ".join([line[1][0] for line in result[0]])
        analysis_results = {}
        
        # Blood Pressure Analysis
        bp_match = re.search(r'(\d{2,3})/(\d{2,3})\s*mm\s*Hg', text_data)
        if bp_match:
            systolic = int(bp_match.group(1))
            diastolic = int(bp_match.group(2))
            status = "Normal" if 120 <= systolic <= 130 and 80 <= diastolic <= 85 else "Abnormal"
            analysis_results['BP'] = f"{systolic}/{diastolic} mmHg ({status})"
            
        # Blood Sugar Analysis
        sugar_match = re.search(r'(\d{2,3})\s*mg/dL', text_data)
        if sugar_match:
            blood_sugar = int(sugar_match.group(1))
            status = "Normal" if 70 <= blood_sugar <= 140 else "Abnormal"
            analysis_results['Blood_Sugar'] = f"{blood_sugar} mg/dL ({status})"
            
        # Heart Rate Analysis
        heart_rate_match = re.search(r'(\d+)\s*bpm', text_data)
        if heart_rate_match:
            heart_rate = int(heart_rate_match.group(1))
            status = "Normal" if 60 <= heart_rate <= 100 else "Abnormal"
            analysis_results['Heart_Rate'] = f"{heart_rate} bpm ({status})"
            
        # Temperature Analysis
        temp_match = re.search(r'(\d+(\.\d+)?)\s*C', text_data)
        if temp_match:
            temperature = float(temp_match.group(1))
            status = "Normal" if 36.5 <= temperature <= 37.5 else "Abnormal"
            analysis_results['Temperature'] = f"{temperature}°C ({status})"
            
        # Oxygen Saturation Analysis
        oxygen_match = re.search(r'(\d+)\s*%', text_data)
        if oxygen_match:
            oxygen = int(oxygen_match.group(1))
            status = "Normal" if 95 <= oxygen <= 100 else "Abnormal"
            analysis_results['Oxygen'] = f"{oxygen}% ({status})"
            
        return analysis_results

def initialize_ai():
    # Initialize the AI model with the API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your environment.")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

# Initialize global objects
keyword_analyzer = MedicalKeywordAnalyzer(os.path.join(CURRENT_DIR, 'data', 'medical_keywords_dataset.csv'))
image_analyzer = MedicalImageAnalyzer()
ai_model = initialize_ai()

def home(request):
    """Render the home page"""
    return render(request, 'home.html')

@csrf_exempt
def analyze(request):
    """Handle medical analysis requests"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST requests are allowed'})
    
    try:
        # Get medical description from form
        description = request.POST.get('description', '')
        
        # Generate boolean query from description
        boolean_query = keyword_analyzer.generate_boolean_query(description)
        
        # Generate AI analysis
        ai_response = ai_model.generate_content(description)
        ai_analysis = ai_response.text
        
        # Handle image analysis if provided
        image_analysis = {}
        if request.FILES.get('image'):
            # Save uploaded image to temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_image:
                temp_image.write(request.FILES['image'].read())
                temp_image_path = temp_image.name
            
            # Analyze image
            try:
                image_analysis = image_analyzer.analyze_health_data(
                    temp_image_path,
                    boolean_query
                )
            finally:
                # Clean up temporary file
                os.unlink(temp_image_path)
        
        # Save analysis results to database
        MedicalAnalysis.objects.create(
            description=description,
            boolean_query=boolean_query,
            ai_analysis=ai_analysis,
            image_analysis=image_analysis
        )
        
        # Return results
        return JsonResponse({
            'success': True,
            'boolean_query': boolean_query,
            'ai_analysis': ai_analysis,
            'image_analysis': image_analysis
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })
