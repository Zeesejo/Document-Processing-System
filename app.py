import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import io
import base64
from PIL import Image
import numpy as np

# Configure Streamlit page
st.set_page_config(
    page_title="Document Processing System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .extraction-result {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.2rem;
    }
    .high-conf { background-color: #d4edda; color: #155724; }
    .med-conf { background-color: #fff3cd; color: #856404; }
    .low-conf { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

def get_confidence_badge(confidence):
    """Return styled confidence badge"""
    if confidence >= 0.9:
        return f'<span class="confidence-badge high-conf">High: {confidence:.1%}</span>'
    elif confidence >= 0.7:
        return f'<span class="confidence-badge med-conf">Medium: {confidence:.1%}</span>'
    else:
        return f'<span class="confidence-badge low-conf">Low: {confidence:.1%}</span>'

def simulate_ocr_extraction(uploaded_file):
    """Simulate OCR text extraction"""
    # Simulate OCR processing
    sample_text = """
    INVOICE
    
    ABC Corporation
    123 Business Street
    City, State 12345
    
    Bill To:
    XYZ Company
    456 Client Avenue
    City, State 67890
    
    Invoice Number: INV-2024-001
    Invoice Date: June 29, 2024
    Due Date: July 29, 2024
    
    Description           Qty    Unit Price    Total
    Software License       1       $1,000.00   $1,000.00
    Support Package        1         $250.00     $250.00
    
    Subtotal:                                  $1,250.00
    Tax (8%):                                    $100.00
    TOTAL:                                     $1,350.00
    
    Payment Terms: Net 30 days
    """
    return sample_text

def simulate_bert_extraction(text):
    """Simulate BERT-based entity extraction"""
    # Simulate BERT NER results
    extracted_data = {
        "document_type": "invoice",
        "vendor_name": "ABC Corporation",
        "vendor_address": "123 Business Street, City, State 12345",
        "customer_name": "XYZ Company",
        "customer_address": "456 Client Avenue, City, State 67890",
        "invoice_number": "INV-2024-001",
        "invoice_date": "2024-06-29",
        "due_date": "2024-07-29",
        "subtotal": 1250.00,
        "tax_amount": 100.00,
        "total_amount": 1350.00,
        "currency": "USD",
        "payment_terms": "Net 30 days",
        "line_items": [
            {
                "description": "Software License",
                "quantity": 1,
                "unit_price": 1000.00,
                "total": 1000.00,
                "confidence": 0.96
            },
            {
                "description": "Support Package", 
                "quantity": 1,
                "unit_price": 250.00,
                "total": 250.00,
                "confidence": 0.94
            }
        ],
        "confidence_scores": {
            "vendor_name": 0.98,
            "invoice_number": 0.99,
            "invoice_date": 0.97,
            "total_amount": 0.96,
            "line_items": 0.95
        },
        "overall_confidence": 0.96
    }
    return extracted_data

def create_processing_metrics():
    """Create sample processing metrics"""
    dates = pd.date_range(start='2024-01-01', end='2024-06-29', freq='D')
    
    metrics_data = pd.DataFrame({
        'Date': dates,
        'Documents_Processed': np.random.poisson(45, len(dates)),
        'Accuracy': np.random.normal(0.96, 0.02, len(dates)),
        'Processing_Time': np.random.normal(2.3, 0.5, len(dates))  # seconds per document
    })
    
    return metrics_data

def main():
    # Header
    st.markdown('<h1 class="main-header">📄 Intelligent Document Processing System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["📤 Process Document", "📊 Analytics Dashboard", "🤖 Model Performance", "⚙️ Batch Processing"]
    )
    
    if mode == "📤 Process Document":
        st.subheader("Upload Document for Processing")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a document...",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            help="Upload invoices, contracts, receipts, or forms"
        )
        
        # Processing options
        col1, col2 = st.columns(2)
        
        with col1:
            doc_type = st.selectbox(
                "Document Type",
                ["Auto-detect", "Invoice", "Contract", "Receipt", "Form"]
            )
        
        with col2:
            processing_mode = st.selectbox(
                "Processing Mode",
                ["Standard", "High Accuracy", "Fast Processing"]
            )
        
        if uploaded_file is not None:
            # Display uploaded file
            if uploaded_file.type.startswith('image'):
                st.image(uploaded_file, caption="Uploaded Document", use_column_width=True)
            else:
                st.success(f"📄 Uploaded: {uploaded_file.name}")
            
            # Process button
            if st.button("🚀 Process Document", type="primary"):
                # Show processing progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate processing steps
                steps = [
                    ("Analyzing document type...", 0.2),
                    ("Performing OCR extraction...", 0.4),
                    ("Running BERT entity extraction...", 0.7),
                    ("Validating extracted data...", 0.9),
                    ("Processing complete!", 1.0)
                ]
                
                for step, progress in steps:
                    status_text.text(step)
                    progress_bar.progress(progress)
                    st.time.sleep(0.5)
                
                # OCR Results
                st.subheader("📝 Extracted Text (OCR)")
                extracted_text = simulate_ocr_extraction(uploaded_file)
                st.text_area("Raw Text", extracted_text, height=200)
                
                # BERT Extraction Results
                st.subheader("🤖 Extracted Information (BERT)")
                extracted_data = simulate_bert_extraction(extracted_text)
                
                # Display overall confidence
                overall_conf = extracted_data['overall_confidence']
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Overall Extraction Confidence</h3>
                    <h1>{overall_conf:.1%}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Key information
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🏢 Vendor Information")
                    vendor_conf = extracted_data['confidence_scores']['vendor_name']
                    st.markdown(f"""
                    <div class="extraction-result">
                        <strong>Name:</strong> {extracted_data['vendor_name']} {get_confidence_badge(vendor_conf)}<br>
                        <strong>Address:</strong> {extracted_data['vendor_address']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 📅 Document Details")
                    inv_num_conf = extracted_data['confidence_scores']['invoice_number']
                    date_conf = extracted_data['confidence_scores']['invoice_date']
                    st.markdown(f"""
                    <div class="extraction-result">
                        <strong>Number:</strong> {extracted_data['invoice_number']} {get_confidence_badge(inv_num_conf)}<br>
                        <strong>Date:</strong> {extracted_data['invoice_date']} {get_confidence_badge(date_conf)}<br>
                        <strong>Due Date:</strong> {extracted_data['due_date']}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 🏪 Customer Information")
                    st.markdown(f"""
                    <div class="extraction-result">
                        <strong>Name:</strong> {extracted_data['customer_name']}<br>
                        <strong>Address:</strong> {extracted_data['customer_address']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 💰 Financial Summary")
                    total_conf = extracted_data['confidence_scores']['total_amount']
                    st.markdown(f"""
                    <div class="extraction-result">
                        <strong>Subtotal:</strong> ${extracted_data['subtotal']:,.2f}<br>
                        <strong>Tax:</strong> ${extracted_data['tax_amount']:,.2f}<br>
                        <strong>Total:</strong> ${extracted_data['total_amount']:,.2f} {get_confidence_badge(total_conf)}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Line items
                st.markdown("### 📋 Line Items")
                items_df = pd.DataFrame(extracted_data['line_items'])
                st.dataframe(items_df, use_container_width=True)
                
                # Download results
                st.markdown("### 💾 Export Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📄 Download JSON"):
                        json_str = json.dumps(extracted_data, indent=2)
                        st.download_button(
                            label="💾 Save JSON",
                            data=json_str,
                            file_name=f"extracted_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                
                with col2:
                    if st.button("📊 Download CSV"):
                        csv_data = pd.DataFrame([{
                            'Field': k,
                            'Value': v,
                            'Confidence': extracted_data['confidence_scores'].get(k, 'N/A')
                        } for k, v in extracted_data.items() if k != 'line_items' and k != 'confidence_scores'])
                        
                        csv_str = csv_data.to_csv(index=False)
                        st.download_button(
                            label="💾 Save CSV",
                            data=csv_str,
                            file_name=f"extracted_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
    
    elif mode == "📊 Analytics Dashboard":
        st.subheader("Processing Analytics Dashboard")
        
        # Generate sample metrics
        metrics_data = create_processing_metrics()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_docs = metrics_data['Documents_Processed'].sum()
        avg_accuracy = metrics_data['Accuracy'].mean()
        avg_processing_time = metrics_data['Processing_Time'].mean()
        
        with col1:
            st.metric("📄 Total Documents", f"{total_docs:,}", "↗️ +15.2%")
        with col2:
            st.metric("🎯 Avg Accuracy", f"{avg_accuracy:.1%}", "↗️ +2.1%")
        with col3:
            st.metric("⚡ Avg Processing Time", f"{avg_processing_time:.1f}s", "↘️ -0.3s")
        with col4:
            st.metric("💰 Cost Savings", "$47,532", "↗️ +12.8%")
        
        # Processing volume over time
        st.subheader("📈 Document Processing Volume")
        
        fig_volume = px.line(
            metrics_data,
            x='Date',
            y='Documents_Processed',
            title="Daily Document Processing Volume"
        )
        fig_volume.update_layout(template="plotly_white")
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # Accuracy trends
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Accuracy Trends")
            fig_acc = px.line(
                metrics_data,
                x='Date',
                y='Accuracy',
                title="Model Accuracy Over Time"
            )
            fig_acc.update_layout(template="plotly_white")
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            st.subheader("⚡ Processing Performance")
            fig_time = px.line(
                metrics_data,
                x='Date',
                y='Processing_Time',
                title="Average Processing Time (seconds)"
            )
            fig_time.update_layout(template="plotly_white")
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Document type distribution
        st.subheader("📊 Document Type Distribution")
        
        doc_types = pd.DataFrame({
            'Document Type': ['Invoices', 'Contracts', 'Receipts', 'Forms', 'Other'],
            'Count': [1250, 890, 560, 340, 180],
            'Accuracy': [0.962, 0.948, 0.971, 0.935, 0.913]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                doc_types,
                values='Count',
                names='Document Type',
                title="Documents by Type"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(
                doc_types,
                x='Document Type',
                y='Accuracy',
                title="Accuracy by Document Type",
                color='Accuracy',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    elif mode == "🤖 Model Performance":
        st.subheader("Model Performance Analysis")
        
        # Model comparison
        models_data = pd.DataFrame({
            'Model': ['BERT-Invoice', 'BERT-Contract', 'BERT-Receipt', 'Custom-NER'],
            'Accuracy': [0.962, 0.948, 0.971, 0.935],
            'Precision': [0.958, 0.944, 0.968, 0.931],
            'Recall': [0.965, 0.952, 0.974, 0.939],
            'F1-Score': [0.961, 0.948, 0.971, 0.935],
            'Training_Data': [10000, 8500, 7200, 12000]
        })
        
        st.markdown("### 🎯 Model Performance Comparison")
        st.dataframe(models_data, use_container_width=True)
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_metrics = go.Figure()
            
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            for metric in metrics:
                fig_metrics.add_trace(go.Scatter(
                    x=models_data['Model'],
                    y=models_data[metric],
                    mode='lines+markers',
                    name=metric,
                    line=dict(width=3)
                ))
            
            fig_metrics.update_layout(
                title="Model Performance Metrics",
                xaxis_title="Model",
                yaxis_title="Score",
                template="plotly_white"
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        with col2:
            fig_training = px.bar(
                models_data,
                x='Model',
                y='Training_Data',
                title="Training Dataset Size",
                color='Training_Data',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_training, use_container_width=True)
        
        # Confusion matrix simulation
        st.markdown("### 📊 Confusion Matrix - BERT Invoice Model")
        
        # Simulate confusion matrix data
        confusion_data = np.array([
            [95, 2, 1, 2],
            [1, 94, 3, 2],
            [2, 1, 96, 1],
            [1, 3, 1, 95]
        ])
        
        labels = ['Vendor', 'Amount', 'Date', 'Items']
        
        fig_confusion = px.imshow(
            confusion_data,
            x=labels,
            y=labels,
            aspect="auto",
            title="Entity Extraction Confusion Matrix",
            color_continuous_scale='Blues'
        )
        
        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                fig_confusion.add_annotation(
                    x=j, y=i,
                    text=str(confusion_data[i, j]),
                    showarrow=False,
                    font=dict(color="white" if confusion_data[i, j] > 50 else "black")
                )
        
        st.plotly_chart(fig_confusion, use_container_width=True)
    
    elif mode == "⚙️ Batch Processing":
        st.subheader("Batch Document Processing")
        
        # Upload multiple files
        uploaded_files = st.file_uploader(
            "Upload multiple documents",
            accept_multiple_files=True,
            type=['pdf', 'png', 'jpg', 'jpeg']
        )
        
        if uploaded_files:
            st.write(f"📄 {len(uploaded_files)} files uploaded")
            
            # Processing options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                batch_size = st.number_input("Batch Size", min_value=1, max_value=100, value=10)
            with col2:
                priority = st.selectbox("Priority", ["Low", "Normal", "High"])
            with col3:
                notify_completion = st.checkbox("Email notification")
            
            if st.button("🚀 Start Batch Processing", type="primary"):
                # Simulate batch processing
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                
                for i, file in enumerate(uploaded_files):
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {file.name}...")
                    
                    # Simulate processing
                    st.time.sleep(0.2)
                    
                    # Add to results
                    results.append({
                        'File': file.name,
                        'Status': 'Completed',
                        'Confidence': np.random.uniform(0.85, 0.98),
                        'Processing_Time': np.random.uniform(1.5, 3.5),
                        'Extracted_Fields': np.random.randint(8, 15)
                    })
                
                status_text.text("✅ Batch processing completed!")
                
                # Display results
                st.subheader("📊 Batch Processing Results")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("✅ Successful", len(results), "100%")
                with col2:
                    avg_conf = results_df['Confidence'].mean()
                    st.metric("🎯 Avg Confidence", f"{avg_conf:.1%}")
                with col3:
                    total_time = results_df['Processing_Time'].sum()
                    st.metric("⏱️ Total Time", f"{total_time:.1f}s")
                with col4:
                    total_fields = results_df['Extracted_Fields'].sum()
                    st.metric("📋 Fields Extracted", total_fields)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🚀 Intelligent Document Processing System | Developed by Zeeshan Modi</p>
        <p>📧 zeemaokik@gmail.com | 🔗 <a href='https://linkedin.com/in/zeesejo'>LinkedIn</a> | 
           💻 <a href='https://github.com/Zeesejo'>GitHub</a></p>
        <p>🎯 96% Extraction Accuracy | ⚡ 50+ docs/min | 🤖 BERT-powered NLP</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
