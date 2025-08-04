import json
import boto3
import numpy as np
from scipy import stats
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import textwrap

class Analysis_Report:
        
### metrics

    def create_performance_summary_tables(self, df, metrics, pdf):
        """Create performance summary tables split across multiple pages if needed."""
        MAX_MODELS_PER_PAGE = 4
        
        # Use all metrics columns except the index columns that were reset
        # Assuming 'model', 'region', and 'inference_profile' are now regular columns
        index_columns = ['model', 'region', 'inference_profile']
        METRICS_TO_SHOW = [col for col in metrics.columns if col not in index_columns]
        
        # Get unique models
        models = metrics['model'].unique()
        model_chunks = [models[i:i + MAX_MODELS_PER_PAGE] for i in range(0, len(models), MAX_MODELS_PER_PAGE)]
        
        for page_num, models_subset in enumerate(model_chunks, 1):
            fig, ax = plt.subplots(figsize=(15, 10))
            plt.axis('off')
            
            data_rows = []
            row_labels = []
            col_labels = []
            valid_columns = []
            print(models_subset)
            
            # Get actual existing combinations from the metrics DataFrame
            for model in models_subset:
                model_display_name = model.split('.')[-1]
                model_data = metrics[metrics['model'] == model]
                
                if not model_data.empty:
                    for _, row in model_data.iterrows():
                        region = row['region']
                        profile = row['inference_profile']
                        
                        # Check if this row has valid data in the metrics columns
                        if not all(pd.isna(row[metric]) for metric in METRICS_TO_SHOW):
                            col_labels.append(f"{model_display_name}\n{region}\n{profile}")
                            valid_columns.append((model, region, profile))
            
            # Create row labels and data only for valid combinations
            for metric in METRICS_TO_SHOW:
                row_labels.append(metric)
                row_data = []
                
                for model, region, profile in valid_columns:
                    # Filter the DataFrame to get the specific row
                    filtered = metrics[(metrics['model'] == model) & 
                                       (metrics['region'] == region) & 
                                       (metrics['inference_profile'] == profile)]
                    
                    if not filtered.empty:
                        value = filtered[metric].iloc[0]  # Get the first (should be only) value
                        
                        if isinstance(value, (int, float)):
                            if metric == 'sample_size':
                                row_data.append(f"{value:.0f}")
                            elif metric == 'avg_cost':                           
                                row_data.append(f"{value:.6f}")
                            else:
                                row_data.append(f"{value:.2f}")
                        else:
                            row_data.append(str(value))
                    else:
                        row_data.append("N/A")  # If combination doesn't exist
                
                data_rows.append(row_data)
            
    
            # Create table only if there are valid columns
            if valid_columns:
                table = ax.table(cellText=data_rows,
                               colLabels=col_labels,
                               rowLabels=row_labels,
                               cellLoc='center',
                               loc='center',
                               bbox=[0.05, 0.05, 0.95, 0.95])
                
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                
                for k, cell in table._cells.items():
                    if k[0] == 0:  # Header row
                        cell.set_height(0.15)
                        cell.set_text_props(ha='center', va='center')
                        cell.set_fontsize(7)
                        cell.set_text_props(weight='bold')
                    
                    if k[1] == -1:  # Row headers (metrics names)
                        cell.set_width(0.20)
                        cell.set_text_props(ha='left')
                    else:
                        cell.set_width(0.80 / len(valid_columns))
                
                plt.title(f'Performance Metrics Summary (Page {page_num} of {len(model_chunks)})', pad=20)
                pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close()
    
    def plot_model_distributions(self, df, metric, metric_name, pdf):
        """Create distribution plots grouped by model, region, and inference profile."""
        model_profiles = df.groupby(['model', 'region', 'inference_profile']).size().reset_index()
        n_combinations = len(model_profiles)
        
        n_cols = min(2, n_combinations)
        n_rows = (n_combinations + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(-1, 1) if n_cols == 1 else axes.reshape(1, -1)
        
        axes_flat = axes.flatten()
        
        for idx, (_, row) in enumerate(model_profiles.iterrows()):
            model = row['model']
            region = row['region']
            profile = row['inference_profile']
            ax = axes_flat[idx]
            
            mask = (df['model'] == model) & (df['region'] == region) & (df['inference_profile'] == profile)
            data = df[mask][metric]
            
            sns.histplot(data=data, kde=True, bins=30, ax=ax)
            
            ax.axvline(data.mean(), color='r', linestyle='--', 
                      label=f'Mean: {data.mean():.2f}')
            ax.axvline(data.median(), color='g', linestyle='--', 
                      label=f'Median: {data.median():.2f}')
            ax.axvline(data.quantile(0.9), color='b', linestyle='--', 
                      label=f'P90: {data.quantile(0.9):.2f}')
            
            model_display_name = model.split('.')[-1]
            ax.set_title(f'{model_display_name}\n{region}\n{profile}')
            ax.set_xlabel(metric_name)
            ax.set_ylabel('Count')
            ax.legend(fontsize='small')
        
        for idx in range(len(model_profiles), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.suptitle(f'{metric_name} Distribution by Model, Region, and Inference Profile')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    def plot_model_comparison(self, df, metric, metric_name, pdf):
        """Create box plot comparing models by inference profile."""
        plt.figure(figsize=(15, 10))
        
        df = df.copy()
        # Create combined model-region display name
        df['model_display'] = df.apply(lambda x: f"{x['model'].split('.')[-1]}\n({x['region']})", axis=1)
        
        # Create box plot with inference_profile as hue
        ax = sns.boxplot(data=df, x='model_display', y=metric, hue='inference_profile')
        
        q1 = df[metric].quantile(0.25)
        q3 = df[metric].quantile(0.75)
        iqr = q3 - q1
        upper_whisker = q3 + 1.5 * iqr
        
        plt.ylim(0, upper_whisker * 1.2)
        plt.title(f'{metric_name} Comparison Across Models and Optimized-Inference')
        plt.xticks(rotation=45)
        plt.legend(title='Profile')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
    def plot_quality_comparison(self, metrics, metric_name, pdf):
        # Create a copy of the dataframe
        df = metrics.copy()
        
        # Create combined model-region display names
        df['model_region'] = df.apply(lambda x: f"{x['model']}\n({x['region']})", axis=1)
        
        # Create the figure
        plt.figure(figsize=(12, 8))
        
        # Create bar chart
        bars = plt.bar(df['model_region'], df[metric_name], alpha=0.7)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Add labels and title
        plt.xlabel('Model (Region)', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(f'{metric_name} Comparison Across Models and Regions', fontsize=14)
        
        # Customize the plot
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, max(df[metric_name]) * 1.1)  # Add some space for the labels
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
    def check_and_create_new_page(self, y_pos, pdf, min_space_needed=0.2):
        """
        Check if we need a new page and create one if necessary.
        Returns: new y_position (either on same or new page)
        """
        if y_pos < min_space_needed:
            pdf.savefig(bbox_inches='tight', dpi=300)
            plt.close()
            
            # Create new page
            fig, ax = plt.subplots(figsize=(12, 12))
            plt.axis('off')
            return 0.95
        return y_pos

        
    def generate_report(self, df, directory, metrics, analysis_summary, quality_evaluation_cost, prompt_optimization_cost) :
        """Main analysis function with PDF report generation."""
        # Turn off interactive plotting
        plt.ioff()
        # Close any existing plots
        plt.close('all')
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_file = os.path.join(f"{directory}-analysis", f'Model_evaluation_analysis_report_{timestamp}.pdf')
        
        with PdfPages(pdf_file) as pdf:
            # Create title page
            fig, ax = plt.subplots(figsize=(12, 12))
            plt.axis('off')
            
            # Main title
            plt.text(0.5, 0.8, 'Model Evaluation Analysis Report',
                    ha='center', va='center', size=24, weight='bold')
            
            # Timestamp
            plt.text(0.5, 0.7, f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                    ha='center', va='center', size=12, style='italic', color='#666666')


            # model section title
            plt.text(0.5, 0.6, 'Models for Evaluation', ha='center', va='center', size=18, weight='bold')
            y_pos = 0.55
            for model in df['model'].unique():
                y_pos = self.check_and_create_new_page(y_pos, pdf)
                model_display_name = model
                plt.text(0.5, y_pos, f"• {model_display_name}", ha='center', va='center', size=12, color='#666666')
                y_pos -= 0.04

            y_pos = y_pos-0.05
            # summary section title
            plt.text(0.5, y_pos, 'Evaluation Summary', ha='center', va='center', size=18, weight='bold')

            y_pos = y_pos-0.05
            # Create your plot with the wrapped text
            #fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed

            width = 70
            # Wrap your text to the desired width (adjust the 70 to your preferred character width)
            wrapped_summary = "\n".join(textwrap.fill(line, width=width) for line in analysis_summary.split('\n'))

            plt.text(0.5, y_pos, wrapped_summary, ha='center', va='top', size=12, color='#666666')
                     #transform=ax.transAxes)  # transform=ax.transAxes helps position text relative to axes


            
            pdf.savefig(bbox_inches='tight', dpi=300)
            plt.close()
            
            #print("Loading data from:", directory)
            #df = combine_csv_files(directory)
            
            # Count error requests
            errored_requests = df[df['api_call_status'] != 'Success']
            errored_count = len(errored_requests)
    
            # Count throttled requests
            throttled_requests = df[df['api_call_status'] == 'ThrottlingException']
            throttled_count = len(throttled_requests)
            
            # Remove error requests from analysis
            df = df[df['api_call_status'] == 'Success']
            
            
            
            # Summary statistics page
            fig, ax = plt.subplots(figsize=(12, 12))
            plt.axis('off')
                   
            # Section 1: API Statistics
            plt.text(0.1, 0.95, 'Summary Statistics', size=18, weight='bold')
            plt.text(0.1, 0.90, f"Total API calls: {len(df) + errored_count}", size=12)
            plt.text(0.1, 0.86, f"Successful calls: {len(df)}", size=12)
            plt.text(0.1, 0.82, f"Errors calls: {errored_count} ({(errored_count/(len(df) + errored_count)*100):.1f}%)", 
                    size=12, color='#666666')
            plt.text(0.1, 0.78, f"Throttled calls: {throttled_count} ({(throttled_count/(len(df) + throttled_count)*100):.1f}%)", 
                    size=12, color='#666666')
            plt.text(0.1, 0.74, f"Model inference cost: {df['cost'].sum():.6f}", size=12)
            plt.text(0.1, 0.70, f"Prompt cptimization cost: {prompt_optimization_cost:.6f}", size=12)
            plt.text(0.1, 0.68, f"LLM-as-a-Judge quality evaluation cost: {quality_evaluation_cost:.6f}", size=12)
            
    
            # Token Statistics section
            plt.text(0.1, 0.60, 'Token Statistics', size=18, weight='bold')
            plt.text(0.1, 0.56, f"Average Input Tokens: {df['model_input_tokens'].mean():.1f}", size=12)
            plt.text(0.1, 0.52, f"Max Input Tokens: {df['model_input_tokens'].max():.0f}", size=12)
            plt.text(0.1, 0.48, f"Average Output Tokens: {df['model_output_tokens'].mean():.1f}", size=12)
            plt.text(0.1, 0.44, f"Max Output Tokens: {df['model_output_tokens'].max():.0f}", size=12)
    
           
            # Section 2: Model Information
            plt.text(0.1, 0.36, 'Model Information', size=18, weight='bold')
            plt.text(0.1, 0.32, f"Number of unique models: {df['model'].nunique()}", size=12)
            plt.text(0.1, 0.28, "Models:", size=12)
            
            y_pos = 0.20
            for model in df['model'].unique():
                y_pos = self.check_and_create_new_page(y_pos, pdf)
                model_display_name = model
                plt.text(0.15, y_pos, f"• {model_display_name}", size=12, color='#2E5A88')
                y_pos -= 0.04
    
            # Section 3: Inference Profiles
            if 'inference_profile' in df.columns:
                y_pos = self.check_and_create_new_page(y_pos, pdf)
                y_pos -= 0.02  # Space between sections
                plt.text(0.1, y_pos, 'Inference Profiles', size=18, weight='bold')
                y_pos -= 0.05
                
                for profile in df['inference_profile'].unique():
                    y_pos = self.check_and_create_new_page(y_pos, pdf)
                    plt.text(0.15, y_pos, f"• {profile}", size=12, color='#2E5A88')
                    y_pos -= 0.04
    
            # Section 4: Sample Distribution
            y_pos = self.check_and_create_new_page(y_pos, pdf)
            y_pos -= 0.02
            plt.text(0.1, y_pos, 'Sample Distribution', size=18, weight='bold')
            y_pos -= 0.05
    
            if 'inference_profile' in df.columns:
                model_profile_counts = df.groupby(['model', 'inference_profile', 'region']).size()
                for (model, profile, region), count in model_profile_counts.items():
                    y_pos = self.check_and_create_new_page(y_pos, pdf)
                    model_display_name = model.split('.')[-1]
                    plt.text(0.15, y_pos, 
                            f"• {model_display_name} in {region} with ({profile}) inference: {count} samples",
                            size=12, color='#2E5A88')
                    y_pos -= 0.04
    
            pdf.savefig(bbox_inches='tight', dpi=300)
            plt.close()
            
            # metrics = calculate_metrics(df, ['model', 'inference_profile'])
            # Basic metrics table
            #metrics = calculate_metrics(df, ['model', 'region', 'inference_profile'])
            self.create_performance_summary_tables(df, metrics, pdf)
    
            #sys.exit()
            #Distribution plots
            self.plot_model_distributions(df, 'latency', 'Latency (seconds)', pdf)
            self.plot_model_distributions(df, 'model_latencyMs', 'Server side latency (ms)', pdf)
            
            # # Model comparisons
            self.plot_model_comparison(df, 'latency', 'Latency (seconds)', pdf)
            self.plot_model_comparison(df, 'model_latencyMs', 'Server side latency (ms)', pdf)
            self.plot_model_comparison(df, 'cost', 'Average Cost per Document Summarization', pdf)
            self.plot_quality_comparison(metrics, 'Builtin.Correctness', pdf)
            
            
            # Save metrics to CSV
            csv_file = os.path.join(f"{directory}-analysis", f'analysis_summary_{timestamp}.csv')
            metrics.to_csv(csv_file)
            
            print(f"\nAnalysis complete!")
            print(f"PDF report saved to: {pdf_file}")
            print(f"CSV summary saved to: {csv_file}")
    