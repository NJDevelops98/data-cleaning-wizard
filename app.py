# ===================================================================
# ========== app.py (The Complete and Final Version) ==========
# ===================================================================

from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from flask_session import Session
import pandas as pd
from io import StringIO, BytesIO
import numpy as np

# Import all cleaning logic functions from your other file
from cleaning_logic import (
    handle_missing_data_master_web,
    handle_duplicates_master_web,
    handle_data_types_master_web,
    handle_categorical_master_web,
    detect_outliers_web,
    treat_outliers_web,
    handle_structural_errors_master_web,
    handle_irrelevant_data_master_web,
    handle_text_formatting_master_web,
    detect_pipeline_issues_interactive,
    apply_pipeline_fixes_interactive
)

# --- App Initialization ---
app = Flask(__name__)
# A secret key is required for flash messages to work
app.config["SECRET_KEY"] = "a_super_secret_key_for_flashing_messages"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# --- Helper Functions (to keep code DRY) ---
def get_df_from_session():
    """Safely retrieves the main DataFrame from the user's session."""
    if 'parquet_data' not in session:
        return None
    try:
        return pd.read_parquet(BytesIO(session['parquet_data']))
    except Exception:
        return None

def save_df_to_session(df):
    """Saves a DataFrame to the user's session as efficient Parquet data."""
    out_buffer = BytesIO()
    df.to_parquet(out_buffer, index=False)
    session['parquet_data'] = out_buffer.getvalue()

# --- Core Routes (Homepage, Upload, Demo, Analysis) ---
@app.route('/')
def home():
    """Renders the homepage."""
    return render_template('index.html')

@app.route('/demo')
def demo():
    """Loads a sample messy dataset to demonstrate the app's features."""
    session.clear()
    try:
        df = pd.read_csv('demo_data.csv')
        save_df_to_session(df)
        session['filename'] = 'demo_data.csv'
        flash("✅ Demo data loaded successfully! You can now explore the cleaning modules.", "success")
        return redirect(url_for('analysis'))
    except FileNotFoundError:
        flash("Error: Demo data file (demo_data.csv) not found in the root directory.", "error")
        return redirect(url_for('home'))
    except Exception as e:
        flash(f"Error loading demo data: {e}", "error")
        return redirect(url_for('home'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handles file uploads from the user."""
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash("No file selected. Please choose a CSV file to upload.", "error")
            return redirect(request.url)
        try:
            session.clear()
            df = pd.read_csv(file, low_memory=False)
            save_df_to_session(df)
            session['filename'] = file.filename
            return redirect(url_for('analysis'))
        except Exception as e:
            flash(f"Error reading file: {e}", "error")
            return redirect(request.url)
    return render_template('upload.html')

@app.route('/analysis')
def analysis():
    """Displays the initial analysis of the uploaded file."""
    df = get_df_from_session()
    if df is None:
        flash("No data found. Please upload a file first.", "error")
        return redirect(url_for('upload'))
    
    column_stats = [{'name': col, 'type': str(df[col].dtype), 'missing': int(df[col].isnull().sum())} for col in df.columns]
    table_html = df.head().to_html(classes='table', index=False, border=0)
    
    return render_template('analysis.html',
                           filename=session.get('filename'),
                           num_rows=len(df),
                           num_cols=len(df.columns),
                           table_html=table_html,
                           column_stats=column_stats)

# --- Manual Module Routes (One-by-one cleaning) ---
@app.route('/modules')
def modules():
    """Displays the main menu for all manual cleaning modules."""
    if get_df_from_session() is None:
        return redirect(url_for('upload'))
    return render_template('modules.html')

@app.route('/module/missing_data')
def module_missing_data_options():
    df = get_df_from_session();
    if df is None: return redirect(url_for('upload'))
    
    blank_rows = df.isnull().all(axis=1).sum()
    total_missing = df.isnull().sum().sum()
    blank_cols = df.columns[df.isnull().all()]
    blank_cols_list, blank_cols_count = blank_cols.tolist(), len(blank_cols)
    
    cols_with_missing = []
    for col in df.columns[(df.isnull().any()) & (~df.isnull().all())]:
        col_info = {'name': col, 'missing': df[col].isnull().sum()}
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({'is_numeric': True, 'mean': df[col].mean(), 'median': df[col].median()})
        else:
            col_info['is_numeric'] = False
        col_info['mode'] = df[col].mode()[0] if not df[col].mode().empty else 'N/A'
        cols_with_missing.append(col_info)
        
    return render_template('module_missing_data.html', total_missing=int(total_missing), blank_rows=int(blank_rows), blank_cols_count=blank_cols_count, blank_cols_list=blank_cols_list, cols_with_missing=cols_with_missing)

@app.route('/module/missing_data/apply', methods=['POST'])
def module_missing_data_action():
    df = get_df_from_session();
    if df is None: return redirect(url_for('upload'))
    df_cleaned = handle_missing_data_master_web(df, request.form)
    save_df_to_session(df_cleaned)
    flash("✅ Successfully applied missing data rules.", 'success')
    return redirect(url_for('modules'))

@app.route('/module/duplicates')
def module_duplicates_options():
    df = get_df_from_session();
    if df is None: return redirect(url_for('upload'))
    return render_template('module_duplicates.html', full_duplicate_count=int(df.duplicated().sum()), all_columns=df.columns.tolist())

@app.route('/module/duplicates/apply', methods=['POST'])
def module_duplicates_action():
    df = get_df_from_session();
    if df is None: return redirect(url_for('upload'))
    df_cleaned = handle_duplicates_master_web(df, request.form)
    save_df_to_session(df_cleaned)
    flash("✅ Successfully applied duplicate handling rules.", 'success')
    return redirect(url_for('modules'))

@app.route('/module/datatypes')
def module_datatypes_options():
    df = get_df_from_session();
    if df is None: return redirect(url_for('upload'))
    return render_template('module_datatypes.html', column_info=[{'name': col, 'type': str(df[col].dtype)} for col in df.columns])

@app.route('/module/datatypes/apply', methods=['POST'])
def module_datatypes_action():
    df = get_df_from_session();
    if df is None: return redirect(url_for('upload'))
    df_cleaned = handle_data_types_master_web(df, request.form)
    save_df_to_session(df_cleaned)
    flash("✅ Successfully applied data type conversions.", 'success')
    return redirect(url_for('modules'))

@app.route('/module/categorical')
def module_categorical_options():
    df = get_df_from_session();
    if df is None: return redirect(url_for('upload'))
    categorical_cols_info = []
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_vals = df[col].dropna().unique()
        if 1 < len(unique_vals) < 75:
            categorical_cols_info.append({'name': col, 'unique_count': len(unique_vals), 'unique_values': sorted(list(unique_vals))})
    return render_template('module_categorical.html', categorical_cols=categorical_cols_info)

@app.route('/module/categorical/apply', methods=['POST'])
def module_categorical_action():
    df = get_df_from_session();
    if df is None: return redirect(url_for('upload'))
    df_cleaned = handle_categorical_master_web(df, request.form)
    save_df_to_session(df_cleaned)
    flash("✅ Successfully applied categorical standardization rules.", 'success')
    return redirect(url_for('modules'))
    
@app.route('/module/outliers')
def module_outliers_options():
    df = get_df_from_session();
    if df is None: return redirect(url_for('upload'))
    return render_template('module_outliers_select_col.html', numeric_cols=df.select_dtypes(include=np.number).columns.tolist())

@app.route('/module/outliers/method', methods=['POST'])
def module_outliers_select_method():
    df = get_df_from_session();
    if df is None: return redirect(url_for('upload'))
    column_name = request.form.get('column_to_analyze')
    return render_template('module_outliers_select_method.html', column_name=column_name, stats_description=df[column_name].describe().to_string())

@app.route('/module/outliers/review', methods=['POST'])
def module_outliers_review():
    df = get_df_from_session();
    if df is None: return redirect(url_for('upload'))
    col_name = request.form.get('column_to_analyze')
    outliers_mask, lower_bound, upper_bound = detect_outliers_web(df, col_name, request.form)
    sample_outliers = df.loc[outliers_mask, col_name].head(5).tolist()
    return render_template('module_outliers_review.html', column_name=col_name, detect_method=request.form.get('detect_method'), outlier_count=outliers_mask.sum(), lower_bound=lower_bound, upper_bound=upper_bound, sample_outliers=sample_outliers)

@app.route('/module/outliers/apply', methods=['POST'])
def module_outliers_apply():
    df = get_df_from_session();
    if df is None: return redirect(url_for('upload'))
    col_name, lower_bound, upper_bound = request.form.get('column_to_analyze'), float(request.form.get('lower_bound')), float(request.form.get('upper_bound'))
    df_cleaned = treat_outliers_web(df, col_name, request.form.get('treat_method'), lower_bound, upper_bound)
    save_df_to_session(df_cleaned)
    flash(f"✅ Successfully applied treatment to column '{col_name}'. Select another column or continue.", 'success'); return redirect(url_for('module_outliers_options'))

@app.route('/module/structural')
def module_structural_options():
    df = get_df_from_session();
    if df is None: return redirect(url_for('upload'))
    blank_rows, blank_cols_list = df.isnull().all(axis=1).sum(), df.columns[df.isnull().all()].tolist()
    return render_template('module_structural.html', blank_rows=int(blank_rows), blank_cols_count=len(blank_cols_list), blank_cols_list=blank_cols_list, num_rows=len(df))

@app.route('/module/structural/apply', methods=['POST'])
def module_structural_action():
    df = get_df_from_session();
    if df is None: return redirect(url_for('upload'))
    df_cleaned = handle_structural_errors_master_web(df, request.form)
    save_df_to_session(df_cleaned)
    flash("✅ Successfully applied structural adjustments.", 'success'); return redirect(url_for('modules'))

@app.route('/module/irrelevant')
def module_irrelevant_options():
    df = get_df_from_session();
    if df is None: return redirect(url_for('upload'))
    return render_template('module_irrelevant.html', all_columns=df.columns.tolist())

@app.route('/module/irrelevant/apply', methods=['POST'])
def module_irrelevant_action():
    df = get_df_from_session();
    if df is None: return redirect(url_for('upload'))
    df_cleaned = handle_irrelevant_data_master_web(df, request.form)
    save_df_to_session(df_cleaned)
    flash("✅ Successfully removed irrelevant data.", 'success'); return redirect(url_for('modules'))

@app.route('/module/formatting')
def module_text_formatting_options():
    df = get_df_from_session();
    if df is None: return redirect(url_for('upload'))
    return render_template('module_formatting.html', text_cols=df.select_dtypes(include=['object']).columns.tolist())

@app.route('/module/formatting/apply', methods=['POST'])
def module_text_formatting_action():
    df = get_df_from_session();
    if df is None: return redirect(url_for('upload'))
    df_cleaned = handle_text_formatting_master_web(df, request.form)
    save_df_to_session(df_cleaned)
    flash("✅ Successfully applied text formatting rules.", 'success'); return redirect(url_for('modules'))

# --- Automatic Pipeline Routes ---
@app.route('/pipeline/run')
def pipeline_run():
    df = get_df_from_session()
    if df is None: return redirect(url_for('upload'))

    findings = detect_pipeline_issues_interactive(df)
    session['pipeline_findings'] = findings
    return render_template('pipeline_summary.html', findings=findings)

@app.route('/pipeline/manual_options')
def pipeline_manual_options():
    if 'pipeline_findings' not in session: return redirect(url_for('pipeline_run'))
    return render_template('pipeline_manual_changes.html', findings=session['pipeline_findings'])

@app.route('/pipeline/apply_defaults', methods=['POST'])
def pipeline_apply_defaults():
    df = get_df_from_session()
    if df is None: return redirect(url_for('upload'))

    default_form = {}
    findings = session.get('pipeline_findings', [])
    for i, finding in enumerate(findings):
        default_form[f'action_type-{i}'] = finding['type']
        default_form[f'column-{i}'] = finding.get('column')
        for option in finding['options']:
            if option.get('is_default'):
                default_form[f'method-{i}'] = option['value']; break
    
    df_cleaned, log = apply_pipeline_fixes_interactive(df, default_form)
    
    out_buffer = BytesIO(); df_cleaned.to_parquet(out_buffer, index=False)
    session['cleaned_parquet_data'] = out_buffer.getvalue() # Store cleaned data separately
    
    return render_template('pipeline_results.html',
                           log=log,
                           num_rows=len(df_cleaned),
                           num_cols=len(df_cleaned.columns),
                           table_html=df_cleaned.head().to_html(classes='table', index=False, border=0))

@app.route('/pipeline/apply_manual', methods=['POST'])
def pipeline_apply_manual():
    df = get_df_from_session()
    if df is None: return redirect(url_for('upload'))

    df_cleaned, log = apply_pipeline_fixes_interactive(df, request.form)

    out_buffer = BytesIO(); df_cleaned.to_parquet(out_buffer, index=False)
    session['cleaned_parquet_data'] = out_buffer.getvalue()

    return render_template('pipeline_results.html',
                           log=log,
                           num_rows=len(df_cleaned),
                           num_cols=len(df_cleaned.columns),
                           table_html=df_cleaned.head().to_html(classes='table', index=False, border=0))

# --- Final Export and Download Routes ---
@app.route('/export')
def export_page():
    """Shows the final data preview and a download button for manually cleaned data."""
    df = get_df_from_session()
    if df is None:
        return redirect(url_for('upload'))
    
    table_html = df.head().to_html(classes='table', index=False, border=0)
    return render_template('export.html',
                           filename=session.get('filename'),
                           num_rows=len(df),
                           num_cols=len(df.columns),
                           table_html=table_html)

@app.route('/download_manual_clean_file')
def download_manual_clean_file():
    """Serves the manually cleaned file for download."""
    df = get_df_from_session()
    if df is None:
        return redirect(url_for('upload'))
    
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    mem_file = BytesIO(csv_buffer.getvalue().encode('utf-8'))
    mem_file.seek(0)
    return send_file(mem_file,
                     as_attachment=True,
                     download_name=f"cleaned_{session.get('filename', 'data.csv')}",
                     mimetype='text/csv')

@app.route('/download_pipeline_file')
def download_pipeline_file():
    """Serves the pipeline-cleaned file for download."""
    if 'cleaned_parquet_data' not in session: return redirect(url_for('upload'))
    
    df = pd.read_parquet(BytesIO(session['cleaned_parquet_data']))
    csv_buffer = StringIO(); df.to_csv(csv_buffer, index=False)
    mem_file = BytesIO(csv_buffer.getvalue().encode('utf-8')); mem_file.seek(0)
    return send_file(mem_file,
                     as_attachment=True,
                     download_name=f"pipeline_cleaned_{session.get('filename', 'data.csv')}",
                     mimetype='text/csv')

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)