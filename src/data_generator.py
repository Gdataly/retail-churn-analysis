import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class RetailChurnDataGenerator:
    """
    Generates realistic retail customer data including:
    - Purchase history
    - Browsing behavior
    - Customer service interactions
    - Return patterns
    - Churn indicators
    """
    
    def __init__(self, start_date='2023-01-01', end_date='2023-12-31', num_customers=1000):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.num_customers = num_customers
        
        # Define customer segments and their churn probabilities
        self.customer_segments = {
            'new_customer': {'prob': 0.20, 'churn_prob': 0.30},
            'loyal': {'prob': 0.35, 'churn_prob': 0.10},
            'at_risk': {'prob': 0.25, 'churn_prob': 0.60},
            'price_sensitive': {'prob': 0.20, 'churn_prob': 0.40}
        }
        
        # Define product categories
        self.categories = {
            'electronics': {'avg_price': 500, 'return_prob': 0.15},
            'clothing': {'avg_price': 100, 'return_prob': 0.20},
            'home_goods': {'avg_price': 200, 'return_prob': 0.10},
            'accessories': {'avg_price': 50, 'return_prob': 0.05},
            'beauty': {'avg_price': 75, 'return_prob': 0.08}
        }
        
        # Customer service interaction types
        self.interaction_types = {
            'complaint': 0.30,
            'inquiry': 0.40,
            'technical_support': 0.20,
            'feedback': 0.10
        }
        
        # Browse behavior patterns
        self.browse_patterns = {
            'high_engagement': {'sessions': (20, 40), 'pages': (10, 30), 'prob': 0.30},
            'medium_engagement': {'sessions': (10, 20), 'pages': (5, 15), 'prob': 0.45},
            'low_engagement': {'sessions': (1, 10), 'pages': (1, 5), 'prob': 0.25}
        }

    def generate_customer_profiles(self):
        """Generate base customer profiles with segments"""
        profiles = []
        
        for customer_id in range(1, self.num_customers + 1):
            # Assign segment based on probabilities
            segment = np.random.choice(
                list(self.customer_segments.keys()),
                p=[seg['prob'] for seg in self.customer_segments.values()]
            )
            
            # Generate registration date
            if segment == 'new_customer':
                reg_date = self.end_date - timedelta(days=random.randint(30, 90))
            else:
                reg_date = self.start_date - timedelta(days=random.randint(0, 365))
            
            # Assign demographic information
            age = np.random.normal(45, 15)  # Age distribution centered at 45
            age = max(18, min(85, int(age)))  # Clip age to realistic range
            
            profile = {
                'customer_id': customer_id,
                'segment': segment,
                'registration_date': reg_date,
                'age': age,
                'gender': np.random.choice(['M', 'F']),
                'location': np.random.choice(['Urban', 'Suburban', 'Rural'], p=[0.45, 0.35, 0.20])
            }
            profiles.append(profile)
        
        return pd.DataFrame(profiles)

    def generate_purchase_history(self, customer_profiles):
        """Generate purchase history for each customer"""
        purchases = []
        
        for _, customer in customer_profiles.iterrows():
            # Determine number of purchases based on segment
            if customer['segment'] == 'loyal':
                n_purchases = random.randint(15, 30)
            elif customer['segment'] == 'new_customer':
                n_purchases = random.randint(1, 5)
            elif customer['segment'] == 'price_sensitive':
                n_purchases = random.randint(5, 15)
            else:  # at_risk
                n_purchases = random.randint(3, 10)
            
            # Generate purchases
            for _ in range(n_purchases):
                purchase_date = pd.date_range(
                    start=customer['registration_date'],
                    end=self.end_date,
                    periods=n_purchases + 1
                )[:-1]
                
                for date in purchase_date:
                    category = np.random.choice(list(self.categories.keys()))
                    base_price = self.categories[category]['avg_price']
                    
                    # Add price variation
                    price = base_price * np.random.normal(1, 0.1)
                    
                    # Determine if returned
                    is_returned = np.random.random() < self.categories[category]['return_prob']
                    
                    purchase = {
                        'customer_id': customer['customer_id'],
                        'date': date,
                        'category': category,
                        'amount': round(price, 2),
                        'is_returned': is_returned
                    }
                    purchases.append(purchase)
        
        return pd.DataFrame(purchases)

    def generate_browsing_behavior(self, customer_profiles):
        """Generate browsing behavior data"""
        browsing = []
        
        for _, customer in customer_profiles.iterrows():
            # Determine engagement level
            engagement = np.random.choice(
                list(self.browse_patterns.keys()),
                p=[p['prob'] for p in self.browse_patterns.values()]
            )
            
            pattern = self.browse_patterns[engagement]
            n_sessions = random.randint(*pattern['sessions'])
            
            for _ in range(n_sessions):
                session_date = pd.date_range(
                    start=customer['registration_date'],
                    end=self.end_date,
                    periods=n_sessions + 1
                )[:-1]
                
                for date in session_date:
                    browsing.append({
                        'customer_id': customer['customer_id'],
                        'date': date,
                        'pages_viewed': random.randint(*pattern['pages']),
                        'session_duration': random.randint(1, 60),
                        'device': np.random.choice(['mobile', 'desktop', 'tablet'], p=[0.45, 0.40, 0.15])
                    })
        
        return pd.DataFrame(browsing)

    def generate_customer_service(self, customer_profiles):
        """Generate customer service interaction data"""
        interactions = []
        
        for _, customer in customer_profiles.iterrows():
            # Number of interactions based on segment
            if customer['segment'] == 'at_risk':
                n_interactions = random.randint(3, 8)
            elif customer['segment'] == 'loyal':
                n_interactions = random.randint(1, 4)
            else:
                n_interactions = random.randint(0, 3)
            
            for _ in range(n_interactions):
                interaction_date = pd.date_range(
                    start=customer['registration_date'],
                    end=self.end_date,
                    periods=n_interactions + 1
                )[:-1]
                
                for date in interaction_date:
                    interaction_type = np.random.choice(
                        list(self.interaction_types.keys()),
                        p=list(self.interaction_types.values())
                    )
                    
                    resolution_time = np.random.exponential(24)  # Hours to resolve
                    satisfaction = None
                    if interaction_type in ['complaint', 'technical_support']:
                        satisfaction = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.3, 0.25, 0.15])
                    
                    interactions.append({
                        'customer_id': customer['customer_id'],
                        'date': date,
                        'interaction_type': interaction_type,
                        'resolution_time_hours': round(resolution_time, 1),
                        'satisfaction_score': satisfaction
                    })
        
        return pd.DataFrame(interactions)

    def generate_churn_labels(self, customer_profiles, purchase_history):
        """Generate churn labels based on customer behavior"""
        last_purchase_dates = purchase_history.groupby('customer_id')['date'].max()
        
        churned = []
        for _, customer in customer_profiles.iterrows():
            # Get last purchase date
            last_purchase = last_purchase_dates.get(customer['customer_id'], customer['registration_date'])
            
            # Calculate days since last purchase
            days_inactive = (self.end_date - pd.to_datetime(last_purchase)).days
            
            # Base churn probability on segment and inactivity
            base_churn_prob = self.customer_segments[customer['segment']]['churn_prob']
            
            # Increase churn probability based on inactivity
            if days_inactive > 90:
                base_churn_prob *= 1.5
            elif days_inactive > 60:
                base_churn_prob *= 1.2
            
            # Determine churn status
            is_churned = np.random.random() < base_churn_prob
            
            churned.append({
                'customer_id': customer['customer_id'],
                'days_inactive': days_inactive,
                'is_churned': is_churned
            })
        
        return pd.DataFrame(churned)

    def generate_complete_dataset(self):
        """Generate complete dataset with all features"""
        print("Generating customer profiles...")
        profiles = self.generate_customer_profiles()
        
        print("Generating purchase history...")
        purchases = self.generate_purchase_history(profiles)
        
        print("Generating browsing behavior...")
        browsing = self.generate_browsing_behavior(profiles)
        
        print("Generating customer service interactions...")
        service = self.generate_customer_service(profiles)
        
        print("Generating churn labels...")
        churn = self.generate_churn_labels(profiles, purchases)
        
        return {
            'customer_profiles': profiles,
            'purchase_history': purchases,
            'browsing_behavior': browsing,
            'customer_service': service,
            'churn_labels': churn
        }

    def save_datasets(self, data, output_dir='data/raw/'):
        """Save all datasets to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in data.items():
            filename = f"{output_dir}{name}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved {filename}")

# Usage example
if __name__ == "__main__":
    generator = RetailChurnDataGenerator()
    data = generator.generate_complete_dataset()
    generator.save_datasets(data)
