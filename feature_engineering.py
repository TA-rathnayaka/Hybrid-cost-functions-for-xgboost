def engineer_credit_features(df, customer_id_col='customer_id', date_col=None):
    """
    Engineer advanced credit risk features from original credit data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with original credit risk features
    customer_id_col : str
        Name of customer identifier column
    date_col : str, optional
        Name of date column for time-based features
        
    Returns:
    --------
    pandas.DataFrame
        Original dataframe with additional engineered features
    """
    df_enhanced = df.copy()
    
    # Sort by customer and date for time-based features
    if date_col and date_col in df.columns:
        df_enhanced = df_enhanced.sort_values([customer_id_col, date_col])
    
    # TIER 1: HIGH IMPACT FEATURES
    
    # 1. Utilization Volatility Score
    if 'P13_REV7140' in df.columns:
        df_enhanced['utilization_volatility_score'] = df_enhanced.groupby(customer_id_col)['P13_REV7140'].transform(
            lambda x: x.rolling(window=min(12, len(x)), min_periods=3).std().fillna(0)
        )
    
    # 2. Minimum Payment Acceleration
    if 'M_PYMT_DUE_MIN_AMT' in df.columns:
        df_enhanced['minimum_payment_acceleration'] = df_enhanced.groupby(customer_id_col)['M_PYMT_DUE_MIN_AMT'].transform(
            lambda x: (x / x.shift(6).fillna(x.iloc[0])) - 1
        ).fillna(0)
    
    # 3. High Utilization + Low Payment Trap
    if all(col in df.columns for col in ['P13_REV7140', 'P13_BRC7160']):
        df_enhanced['high_util_low_payment_trap'] = (
            (df_enhanced['P13_REV7140'] > 80).astype(int) * 
            (df_enhanced['P13_BRC7160'] < 0.05).astype(int)
        )
    
    # 4. Financial Runway Months
    if all(col in df.columns for col in ['D_OEPNTOBUY_CREDITLIMIT', 'M_LSTBILNGSTMNTBLNCE']):
        balance_growth = df_enhanced.groupby(customer_id_col)['M_LSTBILNGSTMNTBLNCE'].transform(
            lambda x: (x - x.shift(3)).fillna(0) / 3
        )
        df_enhanced['financial_runway_months'] = df_enhanced['D_OEPNTOBUY_CREDITLIMIT'] / np.maximum(1, balance_growth)
        df_enhanced['financial_runway_months'] = np.clip(df_enhanced['financial_runway_months'], 0, 999)
    
    # 5. Credit Stress Composite
    if all(col in df.columns for col in ['P13_REV7140', 'P13_REV3424', 'P13_BCX3510', 'FICECLV9_SCORE']):
        util_factor = df_enhanced['P13_REV7140'] / 100
        maxed_ratio = df_enhanced['P13_REV3424'] / np.maximum(1, df_enhanced['P13_BCX3510'])
        score_factor = (90 - df_enhanced['FICECLV9_SCORE'] / 10) / 100
        df_enhanced['credit_stress_composite'] = util_factor * maxed_ratio * np.maximum(0, score_factor)
    
    # TIER 2: BEHAVIORAL SIGNALS
    
    # 6. Balance Reduction Commitment
    if 'M_LSTBILNGSTMNTBLNCE' in df.columns:
        balance_reducing = df_enhanced.groupby(customer_id_col)['M_LSTBILNGSTMNTBLNCE'].transform(
            lambda x: (x < x.shift(1)).astype(int)
        )
        df_enhanced['balance_reduction_commitment'] = balance_reducing.groupby(df_enhanced[customer_id_col]).transform(
            lambda x: x.rolling(window=min(12, len(x)), min_periods=3).mean()
        ).fillna(0)
    
    # 7. Credit Utilization Acceleration
    if 'P13_REV7140' in df.columns:
        df_enhanced['credit_utilization_acceleration'] = df_enhanced.groupby(customer_id_col)['P13_REV7140'].transform(
            lambda x: (x - x.shift(3)) / 3
        ).fillna(0)
    
    # 8. Payment Surge Capacity
    if 'P13_BCC5830' in df.columns:
        df_enhanced['payment_surge_capacity'] = df_enhanced.groupby(customer_id_col)['P13_BCC5830'].transform(
            lambda x: x.rolling(window=min(12, len(x)), min_periods=3).max() / 
                     np.maximum(1, x.rolling(window=min(12, len(x)), min_periods=3).mean())
        ).fillna(1)
    
    # 9. Perfect Score High Stress
    if all(col in df.columns for col in ['FICECLV9_SCORE', 'P13_REV7140']):
        df_enhanced['perfect_score_high_stress'] = (
            (df_enhanced['FICECLV9_SCORE'] > 750).astype(int) * 
            (df_enhanced['P13_REV7140'] > 90).astype(int) * 
            df_enhanced['P13_REV7140']
        )
    
    # 10. Utilization Trajectory 3mo
    if 'P13_REV7140' in df.columns:
        df_enhanced['utilization_trajectory_3mo'] = df_enhanced.groupby(customer_id_col)['P13_REV7140'].transform(
            lambda x: (x - x.shift(3)) / np.maximum(1, x.shift(3))
        ).fillna(0)
    
    # TIER 3: ADVANCED INTERACTIONS
    
    # 11. Payment Timing Consistency
    if 'M_DAYSPASTDUE' in df.columns:
        df_enhanced['payment_timing_consistency'] = df_enhanced.groupby(customer_id_col)['M_DAYSPASTDUE'].transform(
            lambda x: x.rolling(window=min(12, len(x)), min_periods=3).std() / 
                     np.maximum(1, x.rolling(window=min(12, len(x)), min_periods=3).mean())
        ).fillna(0)
    
    # 12. Credit Expansion Restraint
    if 'D_OEPNTOBUY_CREDITLIMIT' in df.columns:
        new_credit = df_enhanced.groupby(customer_id_col)['D_OEPNTOBUY_CREDITLIMIT'].transform(
            lambda x: np.maximum(0, x.shift(1) - x)
        )
        df_enhanced['credit_expansion_restraint'] = new_credit.groupby(df_enhanced[customer_id_col]).transform(
            lambda x: x.rolling(window=6, min_periods=2).sum()
        ) / np.maximum(1, df_enhanced['D_OEPNTOBUY_CREDITLIMIT'])
    
    # 13. Payment to Income Proxy
    if all(col in df.columns for col in ['P13_BCC5830', 'D_OEPNTOBUY_CREDITLIMIT']):
        income_proxy = df_enhanced['D_OEPNTOBUY_CREDITLIMIT'] * 0.02
        df_enhanced['payment_to_income_proxy'] = df_enhanced['P13_BCC5830'] / np.maximum(1, income_proxy)
    
    # 14. Liquidity Buffer Depletion
    if 'D_OEPNTOBUY_CREDITLIMIT' in df.columns:
        df_enhanced['liquidity_buffer_depletion'] = df_enhanced.groupby(customer_id_col)['D_OEPNTOBUY_CREDITLIMIT'].transform(
            lambda x: (x - x.shift(6)) / np.maximum(1, x.shift(6))
        ).fillna(0)
    
    # 15. Delinquency Risk Buildup (weighted recent performance)
    if 'M_DAYSPASTDUE' in df.columns:
        def weighted_delinquency(series):
            weights = np.array([0.5**i for i in range(len(series))])[::-1]
            return np.average(series.fillna(0), weights=weights[-len(series):])
        
        df_enhanced['delinquency_risk_buildup'] = df_enhanced.groupby(customer_id_col)['M_DAYSPASTDUE'].transform(
            lambda x: x.rolling(window=min(6, len(x)), min_periods=2).apply(weighted_delinquency, raw=False)
        ).fillna(0)
    
    # COMPOSITE FEATURES
    
    # 16. Maxed Cards Intensity
    if all(col in df.columns for col in ['P13_BCX3423', 'P13_BCX3510']):
        df_enhanced['maxed_cards_intensity'] = df_enhanced['P13_BCX3423'] / np.maximum(1, df_enhanced['P13_BCX3510'])
    
    # 17. High Utilization Concentration
    if all(col in df.columns for col in ['P13_BCX3422', 'P13_BCX3510']):
        df_enhanced['high_util_concentration'] = df_enhanced['P13_BCX3422'] / np.maximum(1, df_enhanced['P13_BCX3510'])
    
    # 18. Payment Efficiency Ratio
    if all(col in df.columns for col in ['P13_BCC7160', 'P13_REV7140']):
        df_enhanced['payment_efficiency_ratio'] = df_enhanced['P13_BCC7160'] / np.maximum(0.01, df_enhanced['P13_REV7140'] / 100)
    
    # 19. Payment Reliability Score
    if all(col in df.columns for col in ['P13_ALL4520', 'P13_ALL4080']):
        df_enhanced['payment_reliability_score'] = df_enhanced['P13_ALL4520'] / np.maximum(1, df_enhanced['P13_ALL4520'] + df_enhanced['P13_ALL4080'])
    
    # 20. Available Credit Velocity
    if 'D_OEPNTOBUY_CREDITLIMIT' in df.columns:
        df_enhanced['available_credit_velocity'] = df_enhanced.groupby(customer_id_col)['D_OEPNTOBUY_CREDITLIMIT'].transform(
            lambda x: (x - x.shift(6)) / 6
        ).fillna(0)
    
    # 21. Cross Product Stress Contagion
    if all(col in df.columns for col in ['P13_REV7140', 'M_DAYSPASTDUE']):
        credit_stress = df_enhanced['P13_REV7140'] / 100 + df_enhanced['M_DAYSPASTDUE'] / 90
        if 'LOAN_RS_3' in df.columns:
            loan_stress = np.where(df_enhanced['LOAN_RS_3'].notna(), 1 - df_enhanced['LOAN_RS_3'] / 5, 0)
            df_enhanced['cross_product_stress_contagion'] = credit_stress * loan_stress
        else:
            df_enhanced['cross_product_stress_contagion'] = credit_stress
    
    # 22. Anchoring Bias Payments
    if all(col in df.columns for col in ['M_LSTBILNGSTMNTBLNCE', 'P13_BCC5830']):
        def calculate_correlation(group):
            if len(group) < 4:
                return pd.Series([0] * len(group), index=group.index)
            corr = group['M_LSTBILNGSTMNTBLNCE'].corr(group['P13_BCC5830'])
            return pd.Series([1 - abs(corr) if not pd.isna(corr) else 0] * len(group), index=group.index)
        
        df_enhanced['anchoring_bias_payments'] = df_enhanced.groupby(customer_id_col)[
            ['M_LSTBILNGSTMNTBLNCE', 'P13_BCC5830']
        ].apply(calculate_correlation).reset_index(level=0, drop=True)
    
    # 23. Credit Behavior Regime Change
    behavioral_features = ['P13_REV7140', 'P13_BCC5830', 'M_DAYSPASTDUE']
    available_features = [f for f in behavioral_features if f in df.columns]
    
    if available_features:
        regime_changes = []
        for feature in available_features:
            rolling_mean = df_enhanced.groupby(customer_id_col)[feature].transform(
                lambda x: x.rolling(window=min(12, len(x)), min_periods=6).mean()
            )
            rolling_std = df_enhanced.groupby(customer_id_col)[feature].transform(
                lambda x: x.rolling(window=min(12, len(x)), min_periods=6).std()
            )
            z_score = (df_enhanced[feature] - rolling_mean) / np.maximum(0.1, rolling_std)
            regime_changes.append((np.abs(z_score) > 2).astype(int))
        
        df_enhanced['credit_behavior_regime_change'] = np.maximum.reduce(regime_changes) if regime_changes else 0
    
    new_features = len(df_enhanced.columns) - len(df.columns)
    
    
    return df_enhanced

import numpy as np
import pandas as pd

def engineer_fico_features(df, fico_column='fico_credit_score'):
    """
    Generate comprehensive FICO-derived features from input dataframe
    
    Args:
        df: Input pandas DataFrame
        fico_column: Name of FICO score column (default: 'fico_credit_score')
    
    Returns:
        DataFrame with original columns plus engineered FICO features
    """
    result_df = df.copy()
    fico_score = df[fico_column].fillna(300)  # Handle missing values
    
    # Risk tier segmentation
    result_df['fico_tier'] = np.select([
        fico_score >= 800, fico_score >= 740, fico_score >= 670,
        fico_score >= 580, fico_score >= 300
    ], ['exceptional', 'very_good', 'good', 'fair', 'poor'], default='invalid')
    
    # Core transformations
    result_df['fico_percentile'] = np.clip((fico_score - 300) / 550 * 100, 0, 100)
    result_df['fico_creditworthiness'] = ((fico_score - 300) / 550) ** 0.7
    result_df['fico_default_probability'] = np.where(fico_score >= 300, 
                                                   np.exp(-0.015 * (fico_score - 300)), 1.0)
    result_df['fico_approval_likelihood'] = 1 / (1 + np.exp(-0.02 * (fico_score - 650)))
    result_df['fico_stability'] = np.where(fico_score >= 700, 1.0,
                                 np.where(fico_score >= 600, 0.7, 
                                 np.where(fico_score >= 500, 0.4, 0.1)))
    result_df['fico_prime_distance'] = fico_score - 620
    result_df['fico_log_odds'] = -0.025 * fico_score + 15
    result_df['fico_standardized'] = (fico_score - 690) / 65
    result_df['fico_rate_factor'] = 0.10 * np.exp(-0.008 * (fico_score - 500))
    
    # Distance from critical thresholds
    thresholds = [580, 620, 670, 720, 750, 800]
    for threshold in thresholds:
        result_df[f'fico_distance_from_{threshold}'] = fico_score - threshold
    
    # Binning strategies
    result_df['fico_decile'] = np.digitize(fico_score, np.linspace(300, 850, 11)) - 1
    result_df['fico_quintile'] = np.digitize(fico_score, np.linspace(300, 850, 6)) - 1
    result_df['fico_50_bins'] = np.digitize(fico_score, np.linspace(300, 850, 51)) - 1
    
    # Mathematical transformations for interactions
    result_df['fico_squared'] = fico_score ** 2
    result_df['fico_cubed'] = fico_score ** 3
    result_df['log_fico'] = np.log(np.maximum(fico_score, 1))
    result_df['sqrt_fico'] = np.sqrt(np.maximum(fico_score, 0))
    result_df['inverse_fico'] = 1 / np.maximum(fico_score, 1)
    
    # Risk regime binary indicators
    result_df['is_super_prime'] = (fico_score >= 780).astype(int)
    result_df['is_prime'] = ((fico_score >= 660) & (fico_score < 780)).astype(int)
    result_df['is_near_prime'] = ((fico_score >= 620) & (fico_score < 660)).astype(int)
    result_df['is_subprime'] = ((fico_score >= 580) & (fico_score < 620)).astype(int)
    result_df['is_deep_subprime'] = (fico_score < 580).astype(int)
    
    return result_df