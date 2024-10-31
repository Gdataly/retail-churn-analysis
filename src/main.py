```python
# src/main.py

from data_generator import RetailChurnDataGenerator
from churn_analyzer import RetailChurnAnalyzer
import matplotlib.pyplot as plt

def main():
    # 1. Generate Data
    print("Generating retail data...")
    generator = RetailChurnDataGenerator(
        start_date='2023-01-01',
        end_date='2023-12-31',
        num_customers=1000
    )
    data = generator.generate_complete_dataset()
    
    # 2. Run Analysis
    print("\nAnalyzing customer churn...")
    analyzer = RetailChurnAnalyzer()
    
    # Analyze customer behavior
    customer_metrics = analyzer.analyze_customer_behavior(
        data['purchase_history'], 
        data['customer_profiles']
    )
    
    # Identify risks and generate recommendations
    risk_factors = analyzer.identify_risk_factors(customer_metrics)
    recommendations = analyzer.generate_segment_recommendations(customer_metrics)
    action_plan = analyzer.create_action_plan(recommendations)
    
    # Generate summary and visualizations
    summary = analyzer.generate_summary(risk_factors, action_plan)
    fig = analyzer.visualize_insights(customer_metrics, risk_factors, action_plan)
    
    # Print results
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

if __name__ == "__main__":
    main()
```
