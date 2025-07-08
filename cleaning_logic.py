# ===================================================================
# ========== cleaning_logic.py (FINAL, SYNCHRONIZED VERSION) ========
# ===================================================================

import pandas as pd
import numpy as np

# (All the functions from handle_missing_data to handle_text_formatting are correct and included for completeness)
def handle_missing_data_master_web(df, form_data):
    df_cleaned = df.copy()
    if form_data.get('drop_blank_rows') == 'yes': df_cleaned.dropna(how='all', inplace=True)
    if form_data.get('drop_blank_cols') == 'yes': df_cleaned.dropna(axis=1, how='all', inplace=True)
    threshold_str = form_data.get('threshold')
    if threshold_str:
        try:
            pct = float(threshold_str)
            if pct > 0:
                missing_pct = df_cleaned.isnull().mean() * 100
                cols_to_drop = missing_pct[missing_pct > pct].index.tolist()
                if cols_to_drop: df_cleaned.drop(columns=cols_to_drop, inplace=True)
        except (ValueError, TypeError): pass
    for key, method in form_data.items():
        if key.startswith('method-'):
            col_name = key.replace('method-', '')
            if col_name not in df_cleaned.columns: continue
            if method == 'ignore': continue
            if method == 'fill_value':
                fill_val = form_data.get(f'value-{col_name}')
                try: fill_val = float(fill_val)
                except (ValueError, TypeError): pass
                df_cleaned[col_name].fillna(fill_val, inplace=True)
            elif method == 'fill_mean': df_cleaned[col_name].fillna(df_cleaned[col_name].mean(), inplace=True)
            elif method == 'fill_median': df_cleaned[col_name].fillna(df_cleaned[col_name].median(), inplace=True)
            elif method == 'fill_mode':
                mode_val = df_cleaned[col_name].mode()
                if not mode_val.empty: df_cleaned[col_name].fillna(mode_val[0], inplace=True)
            elif method == 'ffill': df_cleaned[col_name].ffill(inplace=True)
            elif method == 'bfill': df_cleaned[col_name].bfill(inplace=True)
    return df_cleaned

def handle_duplicates_master_web(df, form_data):
    df_cleaned = df.copy()
    full_action = form_data.get('full_duplicate_action', 'skip')
    keep_map = {'keep_first': 'first', 'keep_last': 'last', 'drop_all': False}
    if full_action != 'skip':
        if df_cleaned.duplicated().sum() > 0: df_cleaned.drop_duplicates(keep=keep_map[full_action], inplace=True)
    key_columns = form_data.getlist('key_columns')
    if key_columns:
        partial_action = form_data.get('partial_duplicate_action', 'keep_first')
        if df_cleaned.duplicated(subset=key_columns).sum() > 0: df_cleaned.drop_duplicates(subset=key_columns, keep=keep_map[partial_action], inplace=True)
    df_cleaned.reset_index(drop=True, inplace=True)
    return df_cleaned

def handle_data_types_master_web(df, form_data):
    df_cleaned = df.copy()
    for key, new_type in form_data.items():
        if key.startswith('dtype-'):
            col_name = key.replace('dtype-', '')
            if new_type == 'auto' or col_name not in df_cleaned.columns: continue
            try:
                if new_type == 'integer': df_cleaned[col_name] = pd.to_numeric(df_cleaned[col_name], errors='coerce').astype('Int64')
                elif new_type == 'float': df_cleaned[col_name] = pd.to_numeric(df_cleaned[col_name], errors='coerce')
                elif new_type == 'datetime': df_cleaned[col_name] = pd.to_datetime(df_cleaned[col_name], errors='coerce')
                elif new_type == 'text': df_cleaned[col_name] = df_cleaned[col_name].astype(str)
            except Exception: df_cleaned[col_name] = df[col_name]
    return df_cleaned

def handle_categorical_master_web(df, form_data):
    df_cleaned = df.copy()
    cols_to_process = set()
    for key in form_data:
        if key.startswith('trim-'): cols_to_process.add(key.replace('trim-', ''))
        elif key.startswith('case-'): cols_to_process.add(key.replace('case-', ''))
        elif key.startswith('map-'): cols_to_process.add(key.split('-')[1])
    for col_name in cols_to_process:
        if col_name not in df_cleaned.columns or not pd.api.types.is_object_dtype(df_cleaned[col_name]): continue
        working_series = df_cleaned[col_name].copy()
        if form_data.get(f'trim-{col_name}') == 'yes': working_series = working_series.str.strip()
        case_action = form_data.get(f'case-{col_name}')
        if case_action and case_action != 'none':
            if case_action == 'lower': working_series = working_series.str.lower()
            elif case_action == 'upper': working_series = working_series.str.upper()
            elif case_action == 'title': working_series = working_series.str.title()
        mapping_dict = {}
        for key, new_value in form_data.items():
            prefix = f'map-{col_name}-'
            if key.startswith(prefix) and new_value:
                original_value = key[len(prefix):]
                mapping_dict[original_value] = new_value
        if mapping_dict: working_series.replace(mapping_dict, inplace=True)
        df_cleaned[col_name] = working_series
    return df_cleaned

def detect_outliers_web(df, col_name, form_data):
    detect_method = form_data.get('detect_method')
    lower_bound, upper_bound = -np.inf, np.inf
    outliers_mask = pd.Series(False, index=df.index)
    if col_name not in df.columns or not pd.api.types.is_numeric_dtype(df[col_name]): return outliers_mask, lower_bound, upper_bound
    if detect_method == 'iqr':
        Q1, Q3 = df[col_name].quantile(0.25), df[col_name].quantile(0.75)
        IQR = Q3 - Q1
        if pd.notna(IQR) and IQR > 0:
            lower_bound = Q1 - 1.5 * IQR; upper_bound = Q3 + 1.5 * IQR
            outliers_mask = (df[col_name] < lower_bound) | (df[col_name] > upper_bound)
    elif detect_method == 'zscore':
        try:
            z_thresh = float(form_data.get('zscore_thresh', 3.0))
            mean, std = df[col_name].mean(), df[col_name].std()
            if pd.notna(std) and std > 0:
                lower_bound = mean - (z_thresh * std); upper_bound = mean + (z_thresh * std)
                z_scores = np.abs((df[col_name] - mean) / std)
                outliers_mask = z_scores > z_thresh
        except (ValueError, TypeError): pass
    elif detect_method == 'manual':
        try:
            lower_b_str, upper_b_str = form_data.get('lower_bound'), form_data.get('upper_bound')
            lower_bound = float(lower_b_str) if lower_b_str else -np.inf
            upper_bound = float(upper_b_str) if upper_b_str else np.inf
            outliers_mask = (df[col_name] < lower_bound) | (df[col_name] > upper_bound)
        except (ValueError, TypeError): pass
    outliers_mask = outliers_mask & df[col_name].notna()
    return outliers_mask, lower_bound, upper_bound

def treat_outliers_web(df, col_name, treat_method, lower_bound, upper_bound):
    df_cleaned = df.copy()
    outliers_mask = (df_cleaned[col_name] < lower_bound) | (df_cleaned[col_name] > upper_bound)
    outliers_mask = outliers_mask & df_cleaned[col_name].notna()
    if treat_method == 'remove_rows': df_cleaned = df_cleaned[~outliers_mask]
    elif treat_method == 'cap_values': df_cleaned[col_name] = df_cleaned[col_name].clip(lower=lower_bound, upper=upper_bound)
    elif treat_method == 'replace_with_nan': df_cleaned.loc[outliers_mask, col_name] = np.nan
    df_cleaned.reset_index(drop=True, inplace=True)
    return df_cleaned

def handle_structural_errors_master_web(df, form_data):
    df_cleaned = df.copy()
    if form_data.get('drop_blank_rows') == 'yes': df_cleaned.dropna(how='all', inplace=True)
    if form_data.get('drop_blank_cols') == 'yes': df_cleaned.dropna(axis=1, how='all', inplace=True)
    df_cleaned.reset_index(drop=True, inplace=True)
    header_row_str = form_data.get('header_row_index')
    if header_row_str:
        try:
            header_row_index = int(header_row_str)
            if 0 <= header_row_index < len(df_cleaned):
                new_headers = df_cleaned.iloc[header_row_index]
                df_cleaned = df_cleaned.drop(df_cleaned.index[header_row_index])
                df_cleaned.columns = [str(c).strip() if pd.notna(c) else f"Unnamed_Col_{i}" for i, c in enumerate(new_headers)]
                df_cleaned.reset_index(drop=True, inplace=True)
        except (ValueError, TypeError): pass
    rows_to_remove_str = form_data.get('rows_to_remove')
    if rows_to_remove_str:
        indices_to_drop = set()
        parts = rows_to_remove_str.split(',')
        for part in parts:
            part = part.strip()
            try:
                if '-' in part: start, end = map(int, part.split('-')); indices_to_drop.update(range(start, end + 1))
                else: indices_to_drop.add(int(part))
            except ValueError: continue
        valid_indices = [idx for idx in indices_to_drop if 0 <= idx < len(df_cleaned)]
        if valid_indices:
            df_cleaned.drop(index=valid_indices, inplace=True)
            df_cleaned.reset_index(drop=True, inplace=True)
    return df_cleaned

def handle_irrelevant_data_master_web(df, form_data):
    df_cleaned = df.copy()
    cols_to_drop = form_data.getlist('columns_to_drop')
    if cols_to_drop:
        valid_cols_to_drop = [col for col in cols_to_drop if col in df_cleaned.columns]
        if valid_cols_to_drop: df_cleaned.drop(columns=valid_cols_to_drop, inplace=True)
    filter_col, operator, value = form_data.get('filter_column'), form_data.get('filter_operator'), form_data.get('filter_value')
    if filter_col and operator and value and filter_col in df_cleaned.columns:
        col_series = df_cleaned[filter_col]
        if operator in ['gt', 'lt', 'eq', 'neq'] and pd.api.types.is_numeric_dtype(col_series):
            try:
                numeric_val = float(value)
                if operator == 'gt': df_cleaned = df_cleaned[~(col_series > numeric_val)]
                elif operator == 'lt': df_cleaned = df_cleaned[~(col_series < numeric_val)]
                elif operator == 'eq': df_cleaned = df_cleaned[~(col_series == numeric_val)]
                elif operator == 'neq': df_cleaned = df_cleaned[~(col_series != numeric_val)]
            except (ValueError, TypeError): pass
        else:
            col_as_str = col_series.astype(str)
            if operator == 'eq': df_cleaned = df_cleaned[col_as_str != value]
            elif operator == 'neq': df_cleaned = df_cleaned[col_as_str == value]
            elif operator == 'contains': df_cleaned = df_cleaned[~col_as_str.str.contains(value, case=False, na=False)]
            elif operator == 'not_contains': df_cleaned = df_cleaned[col_as_str.str.contains(value, case=False, na=False)]
    return df_cleaned

def handle_text_formatting_master_web(df, form_data):
    df_cleaned = df.copy()
    cols_to_format = form_data.getlist('columns_to_format')
    trim_whitespace = form_data.get('trim_whitespace') == 'yes'
    remove_special = form_data.get('remove_special') == 'yes'
    remove_extra_spaces = form_data.get('remove_extra_spaces') == 'yes'
    for col_name in cols_to_format:
        if col_name in df_cleaned.columns and pd.api.types.is_object_dtype(df_cleaned[col_name]):
            working_series = df_cleaned[col_name].astype(str)
            if trim_whitespace: working_series = working_series.str.strip()
            if remove_extra_spaces: working_series = working_series.str.replace(r'\s+', ' ', regex=True).str.strip()
            if remove_special: working_series = working_series.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
            df_cleaned[col_name] = working_series
    return df_cleaned

# --- THIS IS THE CORRECTED FUNCTION ---
def detect_pipeline_issues_interactive(df):
    findings = []
    
    # Helper to find the default option's text and value
    def get_default_info(options):
        for option in options:
            if option.get('is_default'):
                return option.get('text', ''), option.get('value', None)
        return '', None

    # Finding 1: Blank Rows
    if df.isnull().all(axis=1).any():
        options = [{'value': 'drop', 'text': 'Drop Blank Rows', 'is_default': True}, {'value': 'keep', 'text': 'Keep Blank Rows'}]
        rec_text, def_val = get_default_info(options)
        findings.append({
            'type': 'blank_rows', 'column': None, 
            'description': f"Found {df.isnull().all(axis=1).sum()} fully blank row(s).",
            'recommendation': rec_text, 'default_value': def_val, 'options': options
        })
    
    # Finding 2: Duplicates
    if df.duplicated().any():
        options = [{'value': 'first', 'text': 'Keep First Occurrence', 'is_default': True}, {'value': 'last', 'text': 'Keep Last Occurrence'}, {'value': 'skip', 'text': 'Do Nothing'}]
        rec_text, def_val = get_default_info(options)
        findings.append({
            'type': 'duplicates', 'column': None, 
            'description': f"Found {df.duplicated().sum()} duplicate row(s).",
            'recommendation': rec_text, 'default_value': def_val, 'options': options
        })

    # Finding 3: Missing Values (Column by Column)
    for col in df.columns:
        if df[col].isnull().any():
            description = f"Found {df[col].isnull().sum()} missing values in "
            if pd.api.types.is_numeric_dtype(df[col]):
                options = [{'value': 'fill_median', 'text': 'Fill with Median', 'is_default': True}, {'value': 'fill_mean', 'text': 'Fill with Mean'}, {'value': 'skip', 'text': 'Do Nothing'}]
                description += f"numeric column '{col}'."
                finding_type = 'missing_numeric'
            else:
                options = [{'value': 'ffill', 'text': 'Forward Fill', 'is_default': True}, {'value': 'bfill', 'text': 'Backward Fill'}, {'value': 'fill_mode', 'text': 'Fill with Mode'}, {'value': 'skip', 'text': 'Do Nothing'}]
                description += f"text column '{col}'."
                finding_type = 'missing_text'
            
            rec_text, def_val = get_default_info(options)
            findings.append({
                'type': finding_type, 'column': col,
                'description': description,
                'recommendation': rec_text, 'default_value': def_val, 'options': options
            })
    
    # Finding 4: Whitespace Issues
    text_cols_with_whitespace = [col for col in df.select_dtypes(include=['object']).columns if df[col].astype(str).str.contains(r'^\s|\s$|\s\s', regex=True).any()]
    if text_cols_with_whitespace:
        options = [{'value': 'trim_and_consolidate', 'text': 'Fix Whitespace', 'is_default': True}, {'value': 'skip', 'text': 'Do Nothing'}]
        rec_text, def_val = get_default_info(options)
        findings.append({
            'type': 'whitespace', 'column': ','.join(text_cols_with_whitespace), 
            'description': f"Detected whitespace issues in column(s): {', '.join(text_cols_with_whitespace)}.",
            'recommendation': rec_text, 'default_value': def_val, 'options': options
        })
    
    return findings


def apply_pipeline_fixes_interactive(df, form_data):
    df_cleaned = df.copy()
    log = []
    num_actions = len([key for key in form_data if key.startswith('action_type-')])
    
    for i in range(num_actions):
        action_type = form_data.get(f'action_type-{i}')
        method = form_data.get(f'method-{i}')
        column = form_data.get(f'column-{i}')
        
        if not method or method == 'skip': continue
        
        if action_type == 'blank_rows' and method == 'drop':
            count_before = len(df_cleaned)
            df_cleaned.dropna(how='all', inplace=True)
            if count_before > len(df_cleaned): log.append(f"✔️ Dropped {count_before - len(df_cleaned)} blank rows.")
            
        elif action_type == 'duplicates':
            count_before = len(df_cleaned)
            df_cleaned.drop_duplicates(keep=method, inplace=True)
            if count_before > len(df_cleaned): log.append(f"✔️ Removed {count_before - len(df_cleaned)} duplicates (kept '{method}').")
            
        elif action_type == 'missing_numeric' and column and column in df_cleaned.columns:
            if method == 'fill_mean':
                fill_val = df_cleaned[column].mean()
                df_cleaned[column].fillna(fill_val, inplace=True)
                if pd.notna(fill_val): log.append(f"✔️ Filled missing in '{column}' with mean ({fill_val:.2f}).")
                else: log.append(f"✔️ Attempted to fill '{column}' with mean, but mean could not be calculated.")
            elif method == 'fill_median':
                fill_val = df_cleaned[column].median()
                df_cleaned[column].fillna(fill_val, inplace=True)
                if pd.notna(fill_val): log.append(f"✔️ Filled missing in '{column}' with median ({fill_val:.2f}).")
                else: log.append(f"✔️ Attempted to fill '{column}' with median, but median could not be calculated.")
                
        elif action_type == 'missing_text' and column and column in df_cleaned.columns:
            if method == 'ffill':
                df_cleaned[column].ffill(inplace=True)
                log.append(f"✔️ Forward-filled missing in '{column}'.")
            elif method == 'bfill':
                df_cleaned[column].bfill(inplace=True)
                log.append(f"✔️ Backward-filled missing in '{column}'.")
            elif method == 'fill_mode':
                mode_val = df_cleaned[column].mode()
                if not mode_val.empty:
                    df_cleaned[column].fillna(mode_val[0], inplace=True)
                    log.append(f"✔️ Filled missing in '{column}' with mode ('{mode_val[0]}').")
                    
        elif action_type == 'whitespace' and method == 'trim_and_consolidate' and column:
            affected_cols = column.split(',')
            for col in affected_cols:
                if col in df_cleaned.columns and pd.api.types.is_object_dtype(df_cleaned[col]):
                    df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
            log.append(f"✔️ Fixed whitespace in columns: {column}.")
            
    if not log: log.append("✅ No actions were applied based on your choices.")
    df_cleaned.reset_index(drop=True, inplace=True)
    return df_cleaned, log