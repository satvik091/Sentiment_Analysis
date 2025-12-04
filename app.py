import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import re
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="💭",
    layout="wide"
)

# Title and description
st.title("💭 Sentiment Analysis Tool")
st.markdown("Analyze the sentiment of text using TextBlob's polarity and subjectivity scoring.")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
analysis_mode = st.sidebar.radio(
    "Select Analysis Mode:",
    ["Single Text", "Batch Analysis", "File Upload"]
)

# Helper functions
def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Determine sentiment category
    if polarity > 0.1:
        sentiment = "Positive"
        emoji = "😊"
    elif polarity < -0.1:
        sentiment = "Negative"
        emoji = "😞"
    else:
        sentiment = "Neutral"
        emoji = "😐"
    
    return {
        "text": text,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "sentiment": sentiment,
        "emoji": emoji
    }

def clean_text(text):
    """Basic text cleaning"""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    return text.strip()

def get_word_frequency(text, top_n=10):
    """Get most common words"""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    # Remove common stop words
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'has', 'her', 'was', 'one', 'our', 'out', 'this', 'that', 'with', 'have', 'from'}
    words = [w for w in words if w not in stop_words]
    return Counter(words).most_common(top_n)

# Single Text Analysis
if analysis_mode == "Single Text":
    st.header("📝 Single Text Analysis")
    
    user_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Type or paste your text here..."
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("🔍 Analyze", type="primary")
    with col2:
        clean_check = st.checkbox("Clean text (remove URLs, mentions, hashtags)", value=False)
    
    if analyze_button and user_input:
        # Clean text if requested
        processed_text = clean_text(user_input) if clean_check else user_input
        
        # Analyze sentiment
        result = analyze_sentiment(processed_text)
        
        # Display results
        st.divider()
        st.subheader("Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sentiment", f"{result['emoji']} {result['sentiment']}")
        with col2:
            st.metric("Polarity", f"{result['polarity']:.3f}")
        with col3:
            st.metric("Subjectivity", f"{result['subjectivity']:.3f}")
        
        # Sentiment gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = result['polarity'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment Polarity"},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.1], 'color': "lightcoral"},
                    {'range': [-0.1, 0.1], 'color': "lightyellow"},
                    {'range': [0.1, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation
        st.info("""
        **Understanding the Scores:**
        - **Polarity**: Ranges from -1 (negative) to +1 (positive)
        - **Subjectivity**: Ranges from 0 (objective) to 1 (subjective)
        """)

# Batch Analysis
elif analysis_mode == "Batch Analysis":
    st.header("📊 Batch Text Analysis")
    
    batch_input = st.text_area(
        "Enter multiple texts (one per line):",
        height=200,
        placeholder="Enter each text on a new line..."
    )
    
    if st.button("🔍 Analyze All", type="primary"):
        if batch_input:
            texts = [line.strip() for line in batch_input.split('\n') if line.strip()]
            
            if texts:
                results = [analyze_sentiment(text) for text in texts]
                df = pd.DataFrame(results)
                
                st.divider()
                st.subheader("Summary Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Texts", len(df))
                with col2:
                    positive_count = len(df[df['sentiment'] == 'Positive'])
                    st.metric("Positive", positive_count)
                with col3:
                    neutral_count = len(df[df['sentiment'] == 'Neutral'])
                    st.metric("Neutral", neutral_count)
                with col4:
                    negative_count = len(df[df['sentiment'] == 'Negative'])
                    st.metric("Negative", negative_count)
                
                # Sentiment distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_pie = px.pie(
                        df, 
                        names='sentiment',
                        title='Sentiment Distribution',
                        color='sentiment',
                        color_discrete_map={'Positive': '#90EE90', 'Neutral': '#FFD700', 'Negative': '#FF6B6B'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    fig_box = px.box(
                        df,
                        y='polarity',
                        title='Polarity Distribution',
                        color='sentiment',
                        color_discrete_map={'Positive': '#90EE90', 'Neutral': '#FFD700', 'Negative': '#FF6B6B'}
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # Data table
                st.subheader("Detailed Results")
                display_df = df[['text', 'sentiment', 'polarity', 'subjectivity']].copy()
                display_df['polarity'] = display_df['polarity'].round(3)
                display_df['subjectivity'] = display_df['subjectivity'].round(3)
                st.dataframe(display_df, use_container_width=True)
                
                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )

# File Upload
elif analysis_mode == "File Upload":
    st.header("📁 File Upload Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload a CSV or TXT file",
        type=['csv', 'txt']
    )
    
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1]
        
        if file_type == 'csv':
            df_upload = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df_upload.head())
            
            text_column = st.selectbox(
                "Select the column containing text:",
                df_upload.columns
            )
            
            if st.button("🔍 Analyze", type="primary"):
                with st.spinner("Analyzing..."):
                    results = []
                    for text in df_upload[text_column]:
                        if pd.notna(text):
                            result = analyze_sentiment(str(text))
                            results.append(result)
                    
                    results_df = pd.DataFrame(results)
                    
                    # Merge with original data
                    df_upload['sentiment'] = results_df['sentiment']
                    df_upload['polarity'] = results_df['polarity']
                    df_upload['subjectivity'] = results_df['subjectivity']
                    
                    st.success("Analysis complete!")
                    
                    # Display summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Polarity", f"{results_df['polarity'].mean():.3f}")
                    with col2:
                        st.metric("Most Common", results_df['sentiment'].mode()[0])
                    with col3:
                        st.metric("Avg Subjectivity", f"{results_df['subjectivity'].mean():.3f}")
                    
                    # Visualizations
                    fig = px.histogram(
                        results_df,
                        x='sentiment',
                        title='Sentiment Distribution',
                        color='sentiment',
                        color_discrete_map={'Positive': '#90EE90', 'Neutral': '#FFD700', 'Negative': '#FF6B6B'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show results
                    st.dataframe(df_upload, use_container_width=True)
                    
                    # Download
                    csv = df_upload.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Analyzed Data",
                        data=csv,
                        file_name="analyzed_data.csv",
                        mime="text/csv"
                    )
        
        elif file_type == 'txt':
            text_content = uploaded_file.read().decode('utf-8')
            lines = [line.strip() for line in text_content.split('\n') if line.strip()]
            
            st.write(f"Found {len(lines)} lines of text")
            
            if st.button("🔍 Analyze", type="primary"):
                results = [analyze_sentiment(line) for line in lines]
                df = pd.DataFrame(results)
                
                # Display visualizations and results (similar to batch analysis)
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_pie = px.pie(df, names='sentiment', title='Sentiment Distribution')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    fig_hist = px.histogram(df, x='polarity', title='Polarity Distribution')
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                st.dataframe(df, use_container_width=True)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="sentiment_results.csv",
                    mime="text/csv"
                )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with Streamlit | Powered by TextBlob</p>
</div>
""", unsafe_allow_html=True)
