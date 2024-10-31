import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class RetailChurnAnalyzer:
    """
    A business-focused tool for analyzing and preventing customer churn in retail.
    """
    
    def analyze_customer_behavior(self, transactions_df, customer_df):
        """Analyze key customer behaviors"""
        # Calculate customer metrics
        metrics = transactions_df.groupby('customer_id').agg({
            'transaction_date': lambda x: (pd.Timestamp.now() - pd.to_datetime(x.max())).days,
            'amount': ['count', 'sum', 'mean'],
            'returns': 'sum'
        })
        
        # Flatten column names
        metrics.columns = ['days_since_purchase', 'purchase_count', 
                         'total_spend', 'avg_purchase', 'total_returns']
        metrics = metrics.reset_index()
        
        # Add customer segment info
        customer_metrics = metrics.merge(
            customer_df[['customer_id', 'segment']], 
            on='customer_id',
            how='left'
        )
        
        return customer_metrics
    
    def identify_risk_factors(self, customer_metrics):
        """Identify key risk factors for churn"""
        total_customers = len(customer_metrics)
        risk_factors = []
        
        # 1. Inactivity Risk
        inactive_mask = customer_metrics['days_since_purchase'] > 60
        inactive_count = inactive_mask.sum()
        inactive_customers = customer_metrics[inactive_mask]
        
        risk_factors.append({
            'factor': 'Customer Inactivity (60+ days)',
            'count': int(inactive_count),
            'percentage': np.round((inactive_count / total_customers) * 100, 1),
            'avg_spend': np.round(inactive_customers['total_spend'].mean(), 2)
        })
        
        # 2. High Returns Risk
        returns_ratio = customer_metrics['total_returns'] / customer_metrics['purchase_count']
        high_returns_mask = returns_ratio > 0.3
        high_returns_count = high_returns_mask.sum()
        high_returns = customer_metrics[high_returns_mask]
        
        risk_factors.append({
            'factor': 'High Return Rate (>30%)',
            'count': int(high_returns_count),
            'percentage': np.round((high_returns_count / total_customers) * 100, 1),
            'avg_spend': np.round(high_returns['total_spend'].mean(), 2)
        })
        
        # 3. Declining Purchase Frequency
        avg_purchase_count = customer_metrics['purchase_count'].mean()
        declining_mask = (
            (customer_metrics['purchase_count'] < avg_purchase_count * 0.5) & 
            (customer_metrics['days_since_purchase'] > 30)
        )
        declining_count = declining_mask.sum()
        declining_customers = customer_metrics[declining_mask]
        
        risk_factors.append({
            'factor': 'Declining Purchase Frequency',
            'count': int(declining_count),
            'percentage': np.round((declining_count / total_customers) * 100, 1),
            'avg_spend': np.round(declining_customers['total_spend'].mean(), 2)
        })
        
        return pd.DataFrame(risk_factors)
    
    def generate_segment_recommendations(self, customer_metrics):
        """Generate recommendations for each segment"""
        recommendations = []
        total_customers = len(customer_metrics)
        
        # High-Value Customers at Risk
        high_value_threshold = customer_metrics['total_spend'].quantile(0.75)
        high_value_mask = (
            (customer_metrics['total_spend'] > high_value_threshold) & 
            (customer_metrics['days_since_purchase'] > 30)
        )
        high_value_risk = customer_metrics[high_value_mask]
        
        if len(high_value_risk) > 0:
            recommendations.append({
                'segment': 'High-Value Customers',
                'risk_level': 'High Priority',
                'customer_count': len(high_value_risk),
                'total_revenue_at_risk': np.round(high_value_risk['total_spend'].sum(), 2),
                'action': 'Immediate VIP outreach program',
                'estimated_cost': len(high_value_risk) * 50,  # $50 per customer
                'expected_retention': '70'  # Removed % symbol for calculations
            })
        
        # Medium-Value Customers
        med_value_mask = (
            (customer_metrics['total_spend'].between(
                customer_metrics['total_spend'].quantile(0.4),
                customer_metrics['total_spend'].quantile(0.75)
            )) & 
            (customer_metrics['days_since_purchase'] > 45)
        )
        medium_value_risk = customer_metrics[med_value_mask]
        
        if len(medium_value_risk) > 0:
            recommendations.append({
                'segment': 'Medium-Value Customers',
                'risk_level': 'Medium Priority',
                'customer_count': len(medium_value_risk),
                'total_revenue_at_risk': np.round(medium_value_risk['total_spend'].sum(), 2),
                'action': 'Personalized email campaign with offers',
                'estimated_cost': len(medium_value_risk) * 20,  # $20 per customer
                'expected_retention': '50'
            })
        
        # New Customers at Risk
        new_customer_mask = (
            (customer_metrics['purchase_count'] < 3) & 
            (customer_metrics['days_since_purchase'] > 30)
        )
        new_customer_risk = customer_metrics[new_customer_mask]
        
        if len(new_customer_risk) > 0:
            recommendations.append({
                'segment': 'New Customers',
                'risk_level': 'High Priority',
                'customer_count': len(new_customer_risk),
                'total_revenue_at_risk': np.round(new_customer_risk['total_spend'].sum(), 2),
                'action': 'Welcome back promotion with discount',
                'estimated_cost': len(new_customer_risk) * 30,  # $30 per customer
                'expected_retention': '60'
            })
        
        return pd.DataFrame(recommendations)
    
    def create_action_plan(self, recommendations):
        """Create prioritized action plan with ROI estimates"""
        action_plan = recommendations.copy()
        
        # Calculate potential savings and ROI
        action_plan['potential_savings'] = (
            action_plan['total_revenue_at_risk'] * 
            action_plan['expected_retention'].astype(float) / 100
        )
        
        action_plan['roi'] = np.round(
            (action_plan['potential_savings'] - action_plan['estimated_cost']) /
            action_plan['estimated_cost'] * 100,
            1
        )
        
        return action_plan.sort_values('roi', ascending=False)
    
    def visualize_insights(self, customer_metrics, risk_factors, action_plan):
        """Create business-friendly visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Risk Distribution
        sns.barplot(
            data=risk_factors,
            x='factor',
            y='percentage',
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Risk Factor Distribution')
        axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)
        axes[0, 0].set_ylabel('Percentage of Customers')
        
        # 2. Revenue at Risk by Segment
        sns.barplot(
            data=action_plan,
            x='segment',
            y='total_revenue_at_risk',
            ax=axes[0, 1],
            color='red'
        )
        axes[0, 1].set_title('Revenue at Risk by Segment')
        axes[0, 1].set_ylabel('Revenue ($)')
        
        # 3. ROI of Proposed Actions
        sns.barplot(
            data=action_plan,
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
    
    def generate_summary(self, risk_factors, action_plan):
        """Generate business summary"""
        total_customers_at_risk = risk_factors['count'].sum()
        total_revenue_at_risk = action_plan['total_revenue_at_risk'].sum()
        total_cost = action_plan['estimated_cost'].sum()
        potential_savings = action_plan['potential_savings'].sum()
        
        roi = np.round((potential_savings - total_cost) / total_cost * 100, 1)
        
        summary = {
            'total_customers_at_risk': int(total_customers_at_risk),
            'total_revenue_at_risk': np.round(total_revenue_at_risk, 2),
            'intervention_cost': np.round(total_cost, 2),
            'potential_savings': np.round(potential_savings, 2),
            'roi': roi,
            'priority_actions': action_plan[['segment', 'action', 'estimated_cost', 'roi']].head(3)
        }
        
        return summary

# Example usage
if __name__ == "__main__":
    # Create sample data (one month of transactions)
    np.random.seed(42)
    n_customers = 1000
    n_transactions = 5000
    
    # Generate customer data
    customer_df = pd.DataFrame({
        'customer_id': range(n_customers),
        'segment': np.random.choice(
            ['High Value', 'Medium Value', 'New'], 
            n_customers, 
            p=[0.2, 0.5, 0.3]
        )
    })
    
    # Generate transaction data
    transactions_df = pd.DataFrame({
        'customer_id': np.random.choice(range(n_customers), n_transactions),
        'transaction_date': pd.date_range(
            start='2023-01-01', 
            end='2023-12-31', 
            periods=n_transactions
        ),
        'amount': np.random.normal(100, 30, n_transactions).round(2),
        'returns': np.random.choice([0, 1], n_transactions, p=[0.9, 0.1])
    })
    
    # Run analysis
    analyzer = RetailChurnAnalyzer()
    customer_metrics = analyzer.analyze_customer_behavior(transactions_df, customer_df)
    risk_factors = analyzer.identify_risk_factors(customer_metrics)
    recommendations = analyzer.generate_segment_recommendations(customer_metrics)
    action_plan = analyzer.create_action_plan(recommendations)
    
    # Generate insights
    summary = analyzer.generate_summary(risk_factors, action_plan)
    fig = analyzer.visualize_insights(customer_metrics, risk_factors, action_plan)
    
    # Print key findings
    print("\nKEY FINDINGS AND RECOMMENDATIONS")
    print("================================")
    print(f"Customers at Risk: {summary['total_customers_at_risk']:,}")
    print(f"Revenue at Risk: ${summary['total_revenue_at_risk']:,.2f}")
    print(f"Intervention Cost: ${summary['intervention_cost']:,.2f}")
    print(f"Potential Savings: ${summary['potential_savings']:,.2f}")
    print(f"Expected ROI: {summary['roi']}%")
    
    print("\nPRIORITY ACTIONS:")
    print(summary['priority_actions'])
    
    plt.show()
