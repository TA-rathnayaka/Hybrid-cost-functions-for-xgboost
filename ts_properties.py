from tsfresh.feature_extraction import ComprehensiveFCParameters

custom_tsfresh_dict = {
    # Basic Statistics
    "mean": None,
    "median": None,
    "standard_deviation": None,   # ✅ correct name
    "variance": None,             # ✅ correct name
    "mean_abs_change": None,
    "mean_change": None,
    "mean_second_derivative_central": None,
    "maximum": None,
    "minimum": None,
    "sum_values": None,
    "length": None,
    
    # Percentiles & Quantiles
    "quantile": [{"q": 0.1}, {"q": 0.25}, {"q": 0.75}, {"q": 0.9}],
    "percentile": [{"q": 0.1}, {"q": 0.9}],
    "percentage_of_reoccurring_datapoints_to_all_datapoints": None,
    "percentage_of_reoccurring_values_to_all_values": None,
    
    # Distribution Shape
    "skewness": None,
    "kurtosis": None,
    "absolute_sum_of_changes": None,
    "variation_coefficient": None,
    "variance_larger_than_standard_deviation": None,
    
    # Trend Analysis
    "linear_trend": [{"attr": "slope"}, {"attr": "intercept"}, {"attr": "rvalue"}, {"attr": "stderr"}],
    "agg_linear_trend": [
        {"attr": "slope", "chunk_len": 5, "f_agg": "max"},
        {"attr": "slope", "chunk_len": 5, "f_agg": "min"},
        {"attr": "slope", "chunk_len": 10, "f_agg": "mean"},
    ],
    
    # Peaks & Valleys
    "number_peaks": [{"n": 1}, {"n": 3}, {"n": 5}],
    "number_cwt_peaks": [{"n": 1}, {"n": 5}],
    
    # Autocorrelation & Lag Features
    "autocorrelation": [{"lag": 1}, {"lag": 2}, {"lag": 5}, {"lag": 10}],
    "partial_autocorrelation": [{"lag": 1}, {"lag": 2}, {"lag": 5}],
    "c3": [{"lag": 1}, {"lag": 2}, {"lag": 3}],
    "cid_ce": [{"normalize": True}, {"normalize": False}],
    
    # Frequency Domain
    "fft_coefficient": [
        {"coeff": 0, "attr": "real"}, {"coeff": 0, "attr": "imag"},
        {"coeff": 1, "attr": "real"}, {"coeff": 1, "attr": "imag"},
        {"coeff": 2, "attr": "real"}, {"coeff": 2, "attr": "imag"},
    ],
    "fft_aggregated": [
        {"aggtype": "centroid"}, {"aggtype": "variance"},
        {"aggtype": "skew"}, {"aggtype": "kurtosis"},
    ],
    "spkt_welch_density": [{"coeff": 2}, {"coeff": 5}, {"coeff": 8}],
    "ar_coefficient": [{"coeff": 0, "k": 10}, {"coeff": 1, "k": 10}, {"coeff": 2, "k": 10}],
    
    # Complexity & Regularity
    "sample_entropy": None,
    "approximate_entropy": [{"m": 2, "r": 0.1}],
    
    # Energy & Power
    "abs_energy": None,
    
    # Crossing Features
    "number_crossing_m": [{"m": 0}, {"m": 1}, {"m": -1}],
    "ratio_beyond_r_sigma": [{"r": 0.5}, {"r": 1}, {"r": 1.5}, {"r": 2}],
    
    # Uniqueness & Duplicates
    "has_duplicate_max": None,
    "has_duplicate_min": None,
    "has_duplicate": None,
    
    # Statistical Tests
    "augmented_dickey_fuller": [{"attr": "teststat"}, {"attr": "pvalue"}],
    "benford_correlation": None,
    
    # Histogram & Binning
    "binned_entropy": [{"max_bins": 5}, {"max_bins": 10}],
    "value_count": [{"value": 0}, {"value": 1}, {"value": -1}],
    "range_count": [
        {"min": -2, "max": -1}, {"min": -1, "max": 0},
        {"min": 0, "max": 1}, {"min": 1, "max": 2},
    ],
    
    # Location of Extremes
    "first_location_of_maximum": None,
    "first_location_of_minimum": None,
    "last_location_of_maximum": None,
    "last_location_of_minimum": None,
}
