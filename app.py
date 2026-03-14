import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Richard's Sales Predictor",
    layout="wide"
)

# Load dataset for dropdown options
@st.cache_data
def load_data():
    df = pd.read_csv("options_dataset.csv")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("Wambede_best_model.pkl")
    model_features = joblib.load("model_features.pkl")
    return model, model_features

# Load data and model
try:
    df = load_data()
    model, model_features = load_model()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False
    st.error("Required files not found. Please ensure 'options_dataset.csv', 'Wambede_best_model.pkl', and 'model_features.pkl' are in the current directory.")

# Custom CSS for better styling with light backgrounds
st.markdown("""
<style>
    /* Main container background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(120deg, #ffffff 0%, #f0f4ff 100%);
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 2rem;
        background: rgba(255,255,255,0.7);
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        backdrop-filter: blur(5px);
    }
    
    /* Guide section styling */
    .guide-box {
        background: linear-gradient(145deg, #ffffff 0%, #f8faff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1E3A8A;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
    }
    
    .guide-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
        background: rgba(30, 58, 138, 0.1);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
    }
    
    /* Step boxes in guide */
    .step-box {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
        height: 100%;
        border: 1px solid rgba(30, 58, 138, 0.1);
    }
    
    .step-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(30, 58, 138, 0.2);
    }
    
    /* Prediction box styling */
    .prediction-box {
        background: linear-gradient(145deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 20px 25px -5px rgba(16, 185, 129, 0.3);
    }
    
    .prediction-value {
        font-size: 3.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Input section styling */
    .input-section {
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04);
        border: 1px solid rgba(255,255,255,0.5);
    }
    
    /* Quick tips panel */
    .quick-tips {
        background: linear-gradient(145deg, #ffffff 0%, #f0f7ff 100%);
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid #e0e7ff;
        box-shadow: 0 10px 15px -3px rgba(30, 58, 138, 0.1);
    }
    
    .quick-tips h4 {
        color: #1E3A8A;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e0e7ff;
        padding-bottom: 0.5rem;
    }
    
    .quick-tips ul {
        list-style-type: none;
        padding-left: 0;
    }
    
    .quick-tips li {
        margin-bottom: 0.8rem;
        padding: 0.5rem;
        background: rgba(30, 58, 138, 0.05);
        border-radius: 8px;
        transition: background 0.2s;
    }
    
    .quick-tips li:hover {
        background: rgba(30, 58, 138, 0.1);
    }
    
    /* Model info box */
    .model-info {
        background: linear-gradient(145deg, #e0f2fe 0%, #bae6fd 100%);
        padding: 1rem;
        border-radius: 12px;
        margin-top: 1rem;
        border: 1px solid #7dd3fc;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 20px 20px 0 0;
        border-top: 1px solid #e2e8f0;
        color: #64748b;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(145deg, #1E3A8A 0%, #2563eb 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 10px 15px -3px rgba(30, 58, 138, 0.3);
        transition: all 0.2s;
        font-size: 1.1rem;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 25px -5px rgba(30, 58, 138, 0.4);
        background: linear-gradient(145deg, #2563eb 0%, #1E3A8A 100%);
    }
    
    /* Input field styling */
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        transition: all 0.2s;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #1E3A8A;
        box-shadow: 0 0 0 2px rgba(30, 58, 138, 0.1);
    }
    
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        background-color: white;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #1E3A8A;
        box-shadow: 0 0 0 2px rgba(30, 58, 138, 0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    /* Info boxes */
    .stAlert {
        background: linear-gradient(145deg, #ffffff 0%, #f0f9ff 100%);
        border-radius: 12px;
        border-left: 5px solid #0ea5e9;
    }
    
    /* Placeholder styling */
    .placeholder-box {
        background: linear-gradient(145deg, #ffffff 0%, #f9fafb 100%);
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin-top: 2rem;
        border: 2px dashed #cbd5e1;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Header with light background
st.markdown('<p class="main-header">Retail Sales Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sales forecasting based on product and regional attributes</p>', unsafe_allow_html=True)

# User Guide Section with colored boxes
with st.expander("How to Use This System (Click to expand)", expanded=True):
    st.markdown('<div class="guide-box">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="step-box">
            <h3 style="color: #1E3A8A;">1️⃣ Enter Frequency</h3>
            <p style="color: #4B5563;"><strong>Product Frequency</strong><br>
            How many times product appears<br>
            <span style="background: #e0f2fe; padding: 0.2rem 0.5rem; border-radius: 5px;">Start with 5 if unsure</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="step-box">
            <h3 style="color: #1E3A8A;">2️⃣ Choose Location</h3>
            <p style="color: #4B5563;"><strong>Region → State → City</strong><br>
            Each selection filters the next<br>
            <span style="background: #e0f2fe; padding: 0.2rem 0.5rem; border-radius: 5px;">Dynamic dropdowns</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="step-box">
            <h3 style="color: #1E3A8A;">3️⃣ Select Product</h3>
            <p style="color: #4B5563;"><strong>Category → Sub-Category → Product</strong><br>
            All from historical data<br>
            <span style="background: #e0f2fe; padding: 0.2rem 0.5rem; border-radius: 5px;">Click predict!</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main input section
st.markdown('<p class="guide-title">Input Parameters</p>', unsafe_allow_html=True)

if data_loaded:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        with st.container():
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            
            # Create two columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Product frequency input with help text
                product_frequency = st.number_input(
                    "Product Frequency",
                    min_value=1,
                    max_value=100,
                    value=5,
                    help="Number of times this product appears in historical data."
                )
                
                # Region selection with info
                region = st.selectbox(
                    "Region",
                    options=sorted(df["Region"].unique()),
                    help="Select the geographical region for prediction"
                )
                
                # Filter states by region
                states = sorted(df[df["Region"] == region]["State"].unique())
                state = st.selectbox(
                    "State",
                    options=states,
                    help="Select the state within the chosen region"
                )
                
                # Filter cities by state
                cities = sorted(df[df["State"] == state]["City"].unique())
                city = st.selectbox(
                    "City",
                    options=cities,
                    help="Select the city within the chosen state"
                )
            
            with col2:
                # Category selection
                category = st.selectbox(
                    "Category",
                    options=sorted(df["Category"].unique()),
                    help="Select the product category"
                )
                
                # Filter subcategories
                subcategories = sorted(df[df["Category"] == category]["Sub-Category"].unique())
                sub_category = st.selectbox(
                    "Sub-Category",
                    options=subcategories,
                    help="Select the product sub-category"
                )
                
                # Filter product names
                products = sorted(df[df["Sub-Category"] == sub_category]["Product Name"].unique())
                product_name = st.selectbox(
                    "Product Name",
                    options=products,
                    help="Select the specific product"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        # Quick stats and tips with light background
        st.markdown("""
        <div class="quick-tips">
            <h4>💡 Quick Tips</h4>
            <ul>
                <li>All fields are required</li>
                <li>Options update dynamically</li>
                <li>Predictions are in sales units</li>
                <li>Negative predictions are clipped to 0</li>
            </ul>
            
        </div>
        """.format(len(model_features), type(model).__name__), unsafe_allow_html=True)
    
    # Center the predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("Predict Sales", use_container_width=True)
    
    # Prediction section
    if predict_button:
        # Prepare input data
        input_data = pd.DataFrame({
            "Product_Frequency": [product_frequency],
            "City": [city],
            "State": [state],
            "Region": [region],
            "Category": [category],
            "Sub-Category": [sub_category],
            "Product Name": [product_name]
        })
        
        # Show progress
        with st.spinner("🔄 Analyzing data and generating prediction..."):
            # One-hot encoding
            input_encoded = pd.get_dummies(input_data)
            
            # Align with training features
            input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)
            
            # Predict
            prediction = model.predict(input_encoded)
            
            # Avoid negative predictions
            prediction = np.clip(prediction, 0, None)
        
        # Display prediction in a nice box
        st.markdown(f"""
        <div class="prediction-box">
            <p style="font-size: 1.2rem; margin-bottom: 0.5rem; opacity: 0.95;">Predicted Sales</p>
            <p class="prediction-value">{prediction[0]:.2f}</p>
            <p style="font-size: 1rem; opacity: 0.9;">units</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show input summary
        with st.expander("View Input Summary"):
            summary_data = {
                "Parameter": ["Product Frequency", "Region", "State", "City", "Category", "Sub-Category", "Product Name"],
                "Value": [product_frequency, region, state, city, category, sub_category, product_name]
            }
            summary_df = pd.DataFrame(summary_data)
            st.table(summary_df)
        
        # Confidence note with light background
        st.info("ℹ️ This prediction is based on historical patterns. Actual sales may vary based on market conditions, promotions, and other factors.")
    
    else:
        # Placeholder when no prediction made
        st.markdown("""
        <div class="placeholder-box">
            <p style="color: #6B7280; font-size: 1.2rem;">Fill in the parameters above and click 'Predict Sales' to get started</p>
            <p style="color: #9CA3AF; font-size: 1rem;">Your prediction will appear here</p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("System cannot start due to missing files. Please ensure all required files are present.")

# Footer
st.markdown("""
<div class="footer">
    <p style="font-size: 1rem; font-weight: 500;">Superstore Sales Prediction Application | Developed by Richard Wambede</p>
</div>
""", unsafe_allow_html=True)