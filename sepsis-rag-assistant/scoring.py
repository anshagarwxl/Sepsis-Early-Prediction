# Scoring logic for sepsis-rag-assistant
def calculate_news2(vitals):
    """
    Calculates the National Early Warning Score 2 (NEWS2) based on a dictionary of vital signs.
    Returns the total score and corresponding risk level.
    """
    score = 0
    
    # Safely get vital signs with default values
    respiratory_rate = vitals.get('respiratory_rate', 0)
    spo2 = vitals.get('spo2', 100)
    systolic_bp = vitals.get('systolic_bp', 120)
    heart_rate = vitals.get('heart_rate', 70)
    temperature = vitals.get('temperature', 36.5)
    consciousness = vitals.get('consciousness', 'Alert')

    # [cite_start]Respiratory Rate scoring [cite: 121, 122, 123, 124, 125, 126]
    if respiratory_rate <= 8:
        score += 3
    elif respiratory_rate <= 11:
        score += 1
    elif respiratory_rate >= 21 and respiratory_rate <= 24:
        score += 2
    elif respiratory_rate > 24:
        score += 3
    else:
        score += 0 # Normal range (12-20)

    # [cite_start]SpO2 scoring [cite: 127, 128]
    if spo2 <= 91:
        score += 3
    elif spo2 <= 93:
        score += 2
    elif spo2 <= 95:
        score += 1
    else:
        score += 0

    # [cite_start]Systolic BP scoring [cite: 129, 130, 131, 132, 133]
    if systolic_bp <= 90:
        score += 3
    elif systolic_bp <= 100:
        score += 2
    elif systolic_bp <= 110:
        score += 1
    elif systolic_bp >= 220:
        score += 3
    else:
        score += 0 # Normal range (111-219)

    # [cite_start]Heart Rate scoring [cite: 134, 135, 136, 137, 138, 139, 140, 141, 142]
    if heart_rate <= 40:
        score += 3
    elif heart_rate <= 50:
        score += 1
    elif heart_rate <= 90:
        score += 0
    elif heart_rate <= 110:
        score += 1
    elif heart_rate <= 130:
        score += 2
    else:
        score += 3

    # [cite_start]Temperature scoring [cite: 143, 144, 145, 146, 147, 148, 149]
    if temperature <= 35:
        score += 3
    elif temperature <= 36:
        score += 1
    elif temperature >= 38.1 and temperature <= 39:
        score += 1
    elif temperature > 39:
        score += 2
    else:
        score += 0 # Normal range (36.1-38)

    # [cite_start]Consciousness (AVPU) scoring [cite: 150, 151, 152]
    if consciousness != 'Alert':
        score += 3

    # [cite_start]Risk categorization [cite: 154, 155, 156]
    if score <= 4:
        risk_level = "Low Risk"
    elif score <= 6:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    return score, risk_level

def calculate_qsofa(vitals):
    """
    Calculates the Quick Sequential Organ Failure Assessment (qSOFA) score.
    Returns the total score and corresponding risk level.
    """
    score = 0
    
    # [cite_start]Respiratory rate >= 22 [cite: 161, 162, 163]
    if vitals.get('respiratory_rate', 0) >= 22:
        score += 1
    
    # [cite_start]Altered mental status [cite: 164, 165]
    if vitals.get('consciousness', 'Alert') != 'Alert':
        score += 1
    
    # [cite_start]Systolic BP <= 100 [cite: 166, 167, 168]
    if vitals.get('systolic_bp', 120) <= 100:
        score += 1
        
    # [cite_start]Risk categorization [cite: 169]
    risk_level = "High Risk" if score >= 2 else "Low Risk"
    
    return score, risk_level

def calculate_sirs(vitals):
    """
    Calculates the Systemic Inflammatory Response Syndrome (SIRS) score.
    Returns the total score and corresponding risk level.
    """
    score = 0

    # [cite_start]Temperature > 38 or < 36 [cite: 174, 175, 176, 177, 178]
    temp = vitals.get('temperature', 36.5)
    if temp > 38 or temp < 36:
        score += 1

    # [cite_start]Heart Rate > 90 [cite: 179, 180, 181]
    if vitals.get('heart_rate', 70) > 90:
        score += 1

    # [cite_start]Respiratory Rate > 20 [cite: 182, 183, 184, 185]
    if vitals.get('respiratory_rate', 16) > 20:
        score += 1

    # [cite_start]WBC (if available) [cite: 186, 187, 188, 189]
    wbc = vitals.get('wbc', None)
    if wbc and (wbc > 12000 or wbc < 4000):
        score += 1

    # [cite_start]Risk categorization [cite: 190, 191, 192]
    if score >= 2:
        risk_level = "SIRS Positive"
    else:
        risk_level = "SIRS Negative"

    return score, risk_level

def interpret_risk(news2_score, qsofa_score, sirs_score):
    """
    Provides a comprehensive list of interpretation strings based on the calculated scores.
    """
    interpretations = []

    if news2_score >= 7:
        interpretations.append("A High NEWS2 score indicates urgent clinical assessment needed.")
    elif news2_score >= 5:
        interpretations.append("Medium NEWS2 score suggests increased monitoring required.")

    if qsofa_score >= 2:
        interpretations.append("Positive qSOFA suggests organ dysfunction; sepsis is likely.")
        
    if sirs_score >= 2:
        interpretations.append("SIRS criteria met; systemic inflammatory response is present.")

    return interpretations