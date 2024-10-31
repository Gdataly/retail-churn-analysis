import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class RetailChurnAnalyzer:
    """
    A business-focused tool for analyzing and preventing customer churn in retail.
    Provides clear insights and actionable recommendations for management.
    """
    
    def __init__(self):
        self.risk_segments = {
            'High Risk': {'color': 'red', 'threshold': 0.7},
            'Medium Risk': {'color': 'yellow', 'threshold': 0.4},
            'Low Risk': {'color': 'green', 'threshold': 0}
        }
    
    def analyze_customer_behavior(self, transactions_df, customer_df):
        """
        Analyze key customer behaviors that indicate churn risk
        """
        # Calculate key metrics per customer
        customer_metrics = transactions_df.groupby('customer_id').agg({
            'transaction_date': lambda x: (pd.Timestamp.now() - x.max()).days,  # Days since last purchase
            'amount': ['count', 'sum', 'mean'],
            'returns': 'sum'
        }).round(2)
        
        customer_metrics.columns = [
            'days_since_purchase', 
            'purchase_count', 
            'total_spend', 
            'avg_purchase',
            'total_returns'
        ]
        
        # Add customer segment info
        customer_metrics = customer_metrics.merge(
            customer_df[['customer_id', 'segment']], 
            left_index=True, 
            right_on='customer_id'
        )
        
        return customer_metrics
    
    def identify_risk_factors(self, customer_metrics):
        """
        Identify key risk factors for customer churn
        """
        risk_factors = []
        
        # Inactivity Risk
        inactive_customers = customer_metrics[
            customer_metrics['days_since_purchase'] > 60
        ]
        risk_factors.append({
            'factor': 'Customer Inactivity (60+ days)',
            'count': len(inactive_customers),
            'percentage': (len(inactive_customers) / len(customer_metrics) * 100).round(1),
            'avg_spend': inactive_customers['total_spend'].mean().round(2)
        })
        
        # High Returns Risk
        high_returns = customer_metrics[
            customer_metrics['total_returns'] / customer_metrics['purchase_count'] > 0.3
        ]
        risk_factors.append({
            'factor': 'High Return Rate (>30%)',
            'count': len(high_returns),
            'percentage': (len(high_returns) / len(customer_metrics) * 100).round(1),
            'avg_spend': high_returns['total_spend'].mean().round(2)
        })
        
        # Declining Purchase Frequency
        declining_customers = customer_metrics[
            (customer_metrics['purchase_count'] < customer_metrics['purchase_count'].mean() * 0.5) &
            (customer_metrics['days_since_purchase'] > 30)
        ]
        risk_factors.append({
            'factor': 'Declining Purchase Frequency',
            'count': len(declining_customers),
            'percentage': (len(declining_customers) / len(customer_metrics) * 100).round(1),
            'avg_spend': declining_customers['total_spend'].mean().round(2)
        })
        
        return pd.DataFrame(risk_factors)
    
    def generate_segment_recommendations(self, customer_metrics):
        """
        Generate specific recommendations for each customer segment
        """
        recommendations = []
        
        # High-Value Customers at Risk
        high_value_risk = customer_metrics[
            (customer_metrics['total_spend'] > customer_metrics['total_spend'].quantile(0.75)) &
            (customer_metrics['days_since_purchase'] > 30)
        ]
        
        if len(high_value_risk) > 0:
            recommendations.append({
                'segment': 'High-Value Customers',
                'risk_level': 'High Priority',
                'customer_count': len(high_value_risk),
                'total_revenue_at_risk': high_value_risk['total_spend'].sum().round(2),
                'action': 'Immediate VIP outreach program',
                'estimated_cost': len(high_value_risk) * 50,  # $50 per customer
                'expected_retention': '70%'
            })
        
        # Medium-Value Customers
        medium_value_risk = customer_metrics[
            (customer_metrics['total_spend'].between(
                customer_metrics['total_spend'].quantile(0.4),
                customer_metrics['total_spend'].quantile(0.75)
            )) &
            (customer_metrics['days_since_purchase'] > 45)
        ]
        
        if len(medium_value_risk) > 0:
            recommendations.append({
                'segment': 'Medium-Value Customers',
                'risk_level': 'Medium Priority',
                'customer_count': len(medium_value_risk),
                'total_revenue_at_risk': medium_value_risk['total_spend'].sum().round(2),
                'action': 'Personalized email campaign with offers',
                'estimated_cost': len(medium_value_risk) * 20,  # $20 per customer
                'expected_retention': '50%'
            })
        
        # New Customers at Risk
        new_customer_risk = customer_metrics[
            (customer_metrics['purchase_count'] < 3) &
            (customer_metrics['days_since_purchase'] > 30)
        ]
        
        if len(new_customer_risk) > 0:
            recommendations.append({
                'segment': 'New Customers',
                'risk_level': 'High Priority',
                'customer_count': len(new_customer_risk),
                'total_revenue_at_risk': new_customer_risk['total_spend'].sum().round(2),
                'action': 'Welcome back promotion with discount',
                'estimated_cost': len(new_customer_risk) * 30,  # $30 per customer
                'expected_retention': '60%'
            })
        
        return pd.DataFrame(recommendations)
    
    def create_action_plan(self, recommendations):
        """
        Create a prioritized action plan with ROI estimates
        """
        action_plan = recommendations.copy()
        
        # Calculate potential savings and ROI
        action_plan['potential_savings'] = (
            action_plan['total_revenue_at_risk'] * 
            action_plan['expected_retention'].str.rstrip('%').astype(float) / 100
        )
        
        action_plan['roi'] = (
            (action_plan['potential_savings'] - action_plan['estimated_cost']) /
            action_plan['estimated_cost'] * 100
        ).round(1)
        
        # Sort by ROI
        action_plan = action_plan.sort_values('roi', ascending=False)
        
        return action_plan
    
    def visualize_insights(self, customer_metrics, risk_factors, action_plan):
        """
        Create business-friendly visualizations
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Risk Distribution
        risk_data = pd.DataFrame(risk_factors)
        axes[0, 0].bar(risk_data['factor'], risk_data['percentage'])
        axes[0, 0].set_title('Risk Factor Distribution')
        axes[0, 0].set_xticklabels(risk_data['factor'], rotation=45)
        axes[0, 0].set_ylabel('Percentage of Customers')
        
        # 2. Revenue at Risk by Segment
        revenue_risk = action_plan.plot(
            kind='bar',
            x='segment',
            y='total_revenue_at_risk',
            ax=axes[0, 1],
            color='red'
        )
        axes[0, 1].set_title('Revenue at Risk by Segment')
        axes[0, 1].set_ylabel('Revenue ($)')
        
        # 3. ROI of Proposed Actions
        roi_plot = action_plan.plot(
            kind='bar',
            x='segment',
            y='roi',
            ax=axes[1, 0],
            color='green'
        )
        axes[1, 0].set_title('Expected ROI by Segment')
        axes[1, 0].set_ylabel('ROI (%)')
        
        # 4. Customer Activity Timeline
        sns.histplot(
            data=customer_metrics,
            x='days_since_purchase',
            ax=axes[1, 1],
            bins=30
        )
        axes[1, 1].set_title('Customer Activity Timeline')
        axes[1, 1].set_xlabel('Days Since Last Purchase')
        
        plt.tight_layout()
        return fig
    
    def generate_executive_summary(self, risk_factors, action_plan):
        """
        Generate an executive summary of findings and recommendations
        """
        total_customers_at_risk = risk_factors['count'].sum()
        total_revenue_at_risk = action_plan['total_revenue_at_risk'].sum()
        total_cost = action_plan['estimated_cost'].sum()
        potential_savings = action_plan['potential_savings'].sum()
        
        summary = f"""
        Executive Summary: Customer Churn Risk Analysis
        
        Key Findings:
        -------------
        1. Total Customers at Risk: {total_customers_at_risk:,}
        2. Total Revenue at Risk: ${total_revenue_at_risk:,.2f}
        3. Estimated Intervention Cost: ${total_cost:,.2f}
        4. Potential Revenue Saved: ${potential_savings:,.2f}
        
        Top Priority Actions:
        -------------------
        {self._format_priority_actions(action_plan.head(3))}
        
        Expected Business Impact:
        ----------------------
        - ROI of Recommended Actions: {((potential_savings - total_cost) / total_cost * 100):.1f}%
        - Estimated Timeline: 90 days
        - Resource Requirements: Marketing and Customer Service Teams
        
        Next Steps:
        ----------
        1. Implement high-priority actions immediately
        2. Monitor customer response weekly
        3. Adjust strategies based on response rates
        4. Review progress monthly
        """
        
        return summary
    
    def _format_priority_actions(self, top_actions):
        """Helper function to format priority actions"""
        formatted_actions = ""
        for _, action in top_actions.iterrows():
            formatted_actions += f"""
        * {action['segment']}:
          - Action: {action['action']}
          - Cost: ${action['estimated_cost']:,.2f}
          - Expected ROI: {action['roi']}%
            """
        return formatted_actions

# Example usage
if __name__ == "__main__":
    # Create sample data
    transactions_df = pd.DataFrame({
        'customer_id': range(1000),
        'transaction_date': pd.date_range(start='2023-01-01', end='2023-12-31', periods=1000),
        'amount': np.random.normal(100, 30, 1000),
        'returns': np.random.choice([0, 1], 1000, p=[0.9, 0.1])
    })
    
    customer_df = pd.DataFrame({
        'customer_id': range(1000),
        'segment': np.random.choice(['High Value', 'Medium Value', 'New'], 1000, p=[0.2, 0.5, 0.3])
    })
    
    # Initialize analyzer
    analyzer = RetailChurnAnalyzer()
    
    # Run analysis
    customer_metrics = analyzer.analyze_customer_behavior(transactions_df, customer_df)
    risk_factors = analyzer.identify_risk_factors(customer_metrics)
    recommendations = analyzer.generate_segment_recommendations(customer_metrics)
    action_plan = analyzer.create_action_plan(recommendations)
    
    # Generate visualizations and summary
    fig = analyzer.visualize_insights(customer_metrics, risk_factors, action_plan)
    summary = analyzer.generate_executive_summary(risk_factors, action_plan)
    
    # Print results
    print(summary)
    plt.show()
