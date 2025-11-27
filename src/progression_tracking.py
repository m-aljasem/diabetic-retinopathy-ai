"""
Progression Tracking for Diabetic Retinopathy

Tracks disease progression over time and provides treatment recommendations.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import numpy as np


class RetinopathyProgressionTracker:
    """Track diabetic retinopathy progression."""
    
    def __init__(self):
        """Initialize progression tracker."""
        self.history_file = "data/retinopathy_history.json"
        self.severity_levels = {
            0: 'No DR',
            1: 'Mild',
            2: 'Moderate',
            3: 'Severe',
            4: 'Proliferative'
        }
    
    def add_visit(self, patient_id: str, visit_date: str, severity_level: int, 
                  prediction_data: Dict) -> Dict:
        """
        Add a new visit to patient history.
        
        Args:
            patient_id: Patient identifier
            visit_date: Visit date (YYYY-MM-DD)
            severity_level: DR severity level (0-4)
            prediction_data: Model prediction data
            
        Returns:
            Visit record
        """
        history = self.load_history()
        
        if patient_id not in history:
            history[patient_id] = []
        
        visit = {
            'date': visit_date,
            'severity_level': severity_level,
            'severity_name': self.severity_levels.get(severity_level, 'Unknown'),
            'confidence': prediction_data.get('confidence', 0),
            'probabilities': prediction_data.get('probabilities', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        history[patient_id].append(visit)
        history[patient_id].sort(key=lambda x: x['date'])
        
        self.save_history(history)
        
        return visit
    
    def analyze_progression(self, patient_id: str) -> Dict:
        """
        Analyze progression for a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Progression analysis
        """
        history = self.load_history()
        
        if patient_id not in history or len(history[patient_id]) < 2:
            return {'error': 'Insufficient history for progression analysis'}
        
        visits = history[patient_id]
        
        # Calculate progression metrics
        severity_trend = self._calculate_severity_trend(visits)
        progression_rate = self._calculate_progression_rate(visits)
        risk_of_progression = self._assess_progression_risk(visits)
        
        return {
            'patient_id': patient_id,
            'total_visits': len(visits),
            'first_visit': visits[0]['date'],
            'last_visit': visits[-1]['date'],
            'initial_severity': visits[0]['severity_name'],
            'current_severity': visits[-1]['severity_name'],
            'severity_trend': severity_trend,
            'progression_rate': progression_rate,
            'risk_of_progression': risk_of_progression,
            'recommendations': self._get_progression_recommendations(severity_trend, progression_rate)
        }
    
    def _calculate_severity_trend(self, visits: List[Dict]) -> Dict:
        """Calculate severity trend."""
        severities = [v['severity_level'] for v in visits]
        
        if len(severities) < 2:
            return {'direction': 'stable', 'change': 0}
        
        # Calculate trend
        x = np.arange(len(severities))
        slope = np.polyfit(x, severities, 1)[0]
        
        if slope > 0.1:
            direction = 'worsening'
        elif slope < -0.1:
            direction = 'improving'
        else:
            direction = 'stable'
        
        severity_change = severities[-1] - severities[0]
        
        return {
            'direction': direction,
            'slope': round(slope, 3),
            'change': severity_change,
            'interpretation': self._interpret_trend(direction, severity_change)
        }
    
    def _interpret_trend(self, direction: str, change: int) -> str:
        """Interpret trend."""
        if direction == 'worsening':
            if change >= 2:
                return "Significant progression - urgent treatment consideration"
            elif change == 1:
                return "Moderate progression - treatment may be indicated"
            else:
                return "Mild progression - monitor closely"
        elif direction == 'improving':
            return "Improvement noted - treatment appears effective"
        else:
            return "Stable disease - continue current management"
    
    def _calculate_progression_rate(self, visits: List[Dict]) -> Dict:
        """Calculate progression rate."""
        if len(visits) < 2:
            return {'rate': 0, 'interpretation': 'Insufficient data'}
        
        first_date = datetime.fromisoformat(visits[0]['date'])
        last_date = datetime.fromisoformat(visits[-1]['date'])
        days_diff = (last_date - first_date).days
        
        severity_change = visits[-1]['severity_level'] - visits[0]['severity_level']
        
        if days_diff == 0:
            return {'rate': 0, 'interpretation': 'Same day visits'}
        
        # Rate per year
        rate_per_year = (severity_change / days_diff) * 365
        
        return {
            'rate_per_year': round(rate_per_year, 2),
            'severity_change': severity_change,
            'time_interval_days': days_diff,
            'interpretation': self._interpret_progression_rate(rate_per_year)
        }
    
    def _interpret_progression_rate(self, rate: float) -> str:
        """Interpret progression rate."""
        if rate > 0.5:
            return "Rapid progression - aggressive treatment recommended"
        elif rate > 0.2:
            return "Moderate progression rate"
        elif rate > 0:
            return "Slow progression"
        else:
            return "Stable or improving"
    
    def _assess_progression_risk(self, visits: List[Dict]) -> Dict:
        """Assess risk of future progression."""
        current_severity = visits[-1]['severity_level']
        trend = self._calculate_severity_trend(visits)
        
        # Risk factors
        risk_score = 0
        
        # Current severity
        risk_score += current_severity * 2
        
        # Progression trend
        if trend['direction'] == 'worsening':
            risk_score += 3
        elif trend['direction'] == 'stable':
            risk_score += 1
        
        # Recent progression
        if len(visits) >= 2:
            recent_change = visits[-1]['severity_level'] - visits[-2]['severity_level']
            if recent_change > 0:
                risk_score += 2
        
        # Classify risk
        if risk_score >= 8:
            risk_level = "High"
        elif risk_score >= 5:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'factors': {
                'current_severity': current_severity,
                'trend': trend['direction'],
                'recent_progression': len(visits) >= 2 and visits[-1]['severity_level'] > visits[-2]['severity_level']
            }
        }
    
    def _get_progression_recommendations(self, trend: Dict, progression_rate: Dict) -> List[str]:
        """Get recommendations based on progression."""
        recommendations = []
        
        if trend['direction'] == 'worsening':
            if trend['change'] >= 2:
                recommendations.append("Significant progression - urgent ophthalmology referral")
                recommendations.append("Consider immediate anti-VEGF therapy")
            else:
                recommendations.append("Progression detected - consider treatment")
                recommendations.append("Ophthalmology follow-up within 1-2 months")
        elif trend['direction'] == 'stable':
            current_severity = trend.get('current_severity', 0)
            if current_severity >= 3:
                recommendations.append("Severe disease - continue treatment")
            else:
                recommendations.append("Stable disease - continue monitoring")
        
        if progression_rate.get('rate_per_year', 0) > 0.5:
            recommendations.append("Rapid progression rate - aggressive treatment recommended")
        
        return recommendations
    
    def load_history(self) -> Dict:
        """Load patient history."""
        if Path(self.history_file).exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_history(self, history: Dict):
        """Save patient history."""
        Path(self.history_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)


class ScreeningReminderSystem:
    """Manage screening reminders based on severity."""
    
    def __init__(self):
        """Initialize reminder system."""
        self.screening_intervals = {
            0: 12,  # No DR - annual
            1: 12,  # Mild - annual
            2: 6,   # Moderate - 6 months
            3: 3,   # Severe - 3 months
            4: 1    # Proliferative - monthly
        }
    
    def calculate_next_screening(self, current_severity: int, last_screening_date: str) -> Dict:
        """
        Calculate next screening date.
        
        Args:
            current_severity: Current DR severity level
            last_screening_date: Date of last screening
            
        Returns:
            Next screening information
        """
        interval_months = self.screening_intervals.get(current_severity, 12)
        
        last_date = datetime.fromisoformat(last_screening_date)
        next_date = last_date + timedelta(days=interval_months * 30.44)
        
        days_until = (next_date - datetime.now()).days
        
        return {
            'current_severity': current_severity,
            'recommended_interval_months': interval_months,
            'last_screening': last_screening_date,
            'next_screening_date': next_date.strftime('%Y-%m-%d'),
            'days_until_next': days_until,
            'urgency': 'overdue' if days_until < 0 else 'due_soon' if days_until < 30 else 'scheduled'
        }
    
    def generate_reminder(self, patient_id: str, screening_info: Dict) -> Dict:
        """Generate screening reminder."""
        urgency = screening_info['urgency']
        
        if urgency == 'overdue':
            message = f"OVERDUE: Screening was due {abs(screening_info['days_until_next'])} days ago"
            priority = "High"
        elif urgency == 'due_soon':
            message = f"Screening due in {screening_info['days_until_next']} days"
            priority = "Moderate"
        else:
            message = f"Next screening scheduled for {screening_info['next_screening_date']}"
            priority = "Low"
        
        return {
            'patient_id': patient_id,
            'message': message,
            'priority': priority,
            'next_screening_date': screening_info['next_screening_date'],
            'recommended_interval': screening_info['recommended_interval_months']
        }

