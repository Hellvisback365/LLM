import os
import json
import pandas as pd
from typing import Dict
from datetime import datetime

# ----------------------------
# 1. Experiment Reporter (Mantenuto per ora, valutare spostamento)
# ----------------------------
class ExperimentReporter:
    """Classe per l'analisi e la generazione di report degli esperimenti di raccomandazione."""
    
    def __init__(self, experiments_dir='experiments'):
        """Inizializza il reporter con la directory degli esperimenti."""
        self.experiments_dir = experiments_dir
        os.makedirs(experiments_dir, exist_ok=True)
        self.experiments = []
        self._load_experiments()
    
    def _load_experiments(self):
        """Carica tutti gli esperimenti disponibili nella directory."""
        json_files = [f for f in os.listdir(self.experiments_dir) if f.endswith('.json') and f.startswith('experiment_')]
        for file in json_files:
            try:
                with open(os.path.join(self.experiments_dir, file), 'r', encoding='utf-8') as f:
                    experiment_data = json.load(f)
                    if isinstance(experiment_data, dict) and 'experiment_info' in experiment_data:
                        self.experiments.append(experiment_data)
                    else:
                        print(f"Attenzione: file {file} non sembra un esperimento valido, ignorato.")
            except Exception as e:
                print(f"Errore nel caricamento dell'esperimento {file}: {e}")
    
    def add_experiment(self, experiment_data, filename=None):
        """Aggiunge un nuovo esperimento alla collezione."""
        if isinstance(experiment_data, dict) and 'experiment_info' in experiment_data:
            self.experiments.append(experiment_data)
        else:
            print("Tentativo di aggiungere dati di esperimento non validi.")
            return
        
        if filename:
            target_dir = os.path.dirname(filename)
            # Controlla se il file sorgente è già nella directory degli esperimenti
            if not os.path.abspath(target_dir) == os.path.abspath(self.experiments_dir):
                try:
                    target_path = os.path.join(self.experiments_dir, os.path.basename(filename))
                    if os.path.exists(filename):
                        with open(filename, 'r', encoding='utf-8') as src_f, \
                                open(target_path, 'w', encoding='utf-8') as dst_f:
                            json.dump(json.load(src_f), dst_f, ensure_ascii=False, indent=2)
                        print(f"Esperimento copiato in {target_path}")
                    else:
                        print(f"File sorgente {filename} non trovato per la copia.")
                except Exception as e:
                    print(f"Errore nel copiare/salvare il file esperimento in {self.experiments_dir}: {e}")
            # else: # Opzionale: gestisci il caso in cui il file è già nella dir giusta
            #    print(f"Il file {filename} è già nella directory degli esperimenti.")
    
    def calculate_diversity_metrics(self):
        """Calcola metriche di diversità tra gli esperimenti."""
        if not self.experiments:
            return {"error": "Nessun esperimento disponibile"}
        
        metrics = {
            "total_experiments": len(self.experiments),
            "unique_final_recommendations": set(),
            "recommendation_frequency": {},
            "metric_performance": {}
        }
        all_final_recommendations = []
        
        for exp in self.experiments:
            final_eval = exp.get('final_evaluation', {})
            final_recs = final_eval.get('final_recommendations', [])
            if final_recs and isinstance(final_recs, list):
                all_final_recommendations.extend(final_recs)
                for rec in final_recs:
                    # Assicura che rec sia hashabile (es. int)
                    try:
                        hashable_rec = int(rec)
                        metrics["recommendation_frequency"][hashable_rec] = metrics["recommendation_frequency"].get(hashable_rec, 0) + 1
                    except (ValueError, TypeError):
                        print(f"Attenzione: Impossibile usare la raccomandazione '{rec}' come chiave di frequenza.")
            
            metric_recs_per_exp = exp.get('metric_recommendations', {})
            for user_id, user_data in metric_recs_per_exp.items():
                 if isinstance(user_data, dict):
                      for metric_name, metric_data in user_data.items():
                           if isinstance(metric_data, dict):
                                if metric_name not in metrics["metric_performance"]:
                                    metrics["metric_performance"][metric_name] = {
                        "total_recommendations": [],
                        "explanation_themes": {}
                    }
                                recs = metric_data.get('recommendations', [])
                                if recs and isinstance(recs, list):
                                     # Aggiungi solo recs hashabili
                                     hashable_recs = []
                                     for r in recs:
                                         try: hashable_recs.append(int(r))
                                         except (ValueError, TypeError): pass
                                     metrics["metric_performance"][metric_name]["total_recommendations"].extend(hashable_recs)
                                
                                explanation = metric_data.get('explanation', '').lower()
                                themes = []
                                if "popolare" in explanation or "popolarità" in explanation: themes.append("popolarità")
                                if "qualità" in explanation: themes.append("qualità")
                                if "diversità" in explanation or "diversi" in explanation: themes.append("diversità")
                                if "genere" in explanation or "generi" in explanation: themes.append("generi")
                                for theme in themes:
                                    theme_counts = metrics["metric_performance"][metric_name]["explanation_themes"]
                                    theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        # Converti set in lista ordinata per JSON
        unique_final_rec_list = sorted([int(r) for r in all_final_recommendations if isinstance(r, (int, float)) or str(r).isdigit()])
        metrics["unique_final_recommendations"] = {
            "count": len(set(unique_final_rec_list)),
            "items": sorted(list(set(unique_final_rec_list)))
        }
        
        for metric, data in metrics["metric_performance"].items():
            # Assicura che tutte le raccomandazioni siano interi prima di creare il set
            valid_recs = [int(r) for r in data["total_recommendations"] if isinstance(r, (int, float)) or str(r).isdigit()]
            unique_metric_recs = set(valid_recs)
            data["unique_recommendations"] = {
                "count": len(unique_metric_recs),
                "items": sorted(list(unique_metric_recs))
            }
            del data["total_recommendations"] # Rimuovi lista grezza
             
        metrics["recommendation_frequency"] = dict(sorted(metrics["recommendation_frequency"].items(), key=lambda item: item[1], reverse=True))
        
        return metrics
    
    def generate_html_report(self, output_file="experiment_report.html") -> str:
        """Genera un report HTML dettagliato degli esperimenti."""
        if not self.experiments:
            return "Nessun esperimento disponibile per generare il report."
        
        movies_df = None
        try:
             movies_path = os.path.join('data', 'processed', 'movies.csv')
             if os.path.exists(movies_path):
                  movies_df = pd.read_csv(movies_path, index_col='movie_id')
        except Exception as e:
             print(f"Attenzione: impossibile caricare movies.csv: {e}")

        def get_movie_title(movie_id):
            try:
                mid_int = int(movie_id)
                if movies_df is not None and mid_int in movies_df.index:
                    return movies_df.loc[mid_int, 'title']
            except (ValueError, TypeError, KeyError):
                 pass
            return f"ID: {movie_id}" # Fallback
            
        diversity_metrics = self.calculate_diversity_metrics()
        # CSS più compatto
        html = """<!DOCTYPE html><html><head><title>Report Esperimenti</title><style>
            body{font-family: sans-serif; margin: 20px; line-height: 1.5;} 
            h1, h2, h3{color: #333;} table{border-collapse: collapse; width: 100%; margin-bottom: 20px;} 
            th, td{border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top;} 
            th{background-color: #f2f2f2;} 
            .experiment{margin-bottom: 30px; border: 1px solid #ccc; padding: 15px; border-radius: 5px; background: #f9f9f9;} 
            pre{background-color: #eee; padding: 10px; border-radius: 3px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word;} 
            .metrics-container{display: flex; flex-wrap: wrap; gap: 20px;} 
            .metric-block{flex: 1; min-width: 250px; margin-bottom: 10px;} 
        </style></head><body>"""
        html += f"<h1>Report Esperimenti ({datetime.now().strftime('%Y-%m-%d %H:%M')})</h1>"
        
        html += "<h2>Statistiche Generali</h2><table>"
        html += f"<tr><th>Totale Esperimenti</th><td>{diversity_metrics['total_experiments']}</td></tr>"
        unique_final = diversity_metrics['unique_final_recommendations']
        html += f"<tr><th>Raccomandazioni Finali Uniche</th><td>{unique_final.get('count', 0)}</td></tr>"
        html += "</table>"
        
        html += "<h3>Film più Raccomandati (Finali)</h3><table><tr><th>Film</th><th>Frequenza</th></tr>"
        freq_items = list(diversity_metrics["recommendation_frequency"].items())[:15]
        for movie_id, frequency in freq_items:
            html += f"<tr><td>{get_movie_title(movie_id)}</td><td>{frequency}</td></tr>"
        html += "</table>"
        
        html += "<h2>Performance per Metrica (Aggregata)</h2>"
        for metric, data in diversity_metrics.get("metric_performance", {}).items():
            html += f"<div class='metric-block'><h3>Metrica: {metric}</h3><table>"
            unique_metric = data.get("unique_recommendations", {})
            html += f"<tr><th>Raccomandazioni Uniche ({metric})</th><td>{unique_metric.get('count', 0)}</td></tr></table>"
            themes = data.get("explanation_themes", {})
            if themes:
                html += "<h4>Temi nelle Spiegazioni</h4><table><tr><th>Tema</th><th>Freq.</th></tr>"
                sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
            for theme, frequency in sorted_themes:
                html += f"<tr><td>{theme}</td><td>{frequency}</td></tr>"
            html += "</table>"
            html += "</div>"

        html += "<h2>Dettagli Esperimenti</h2>"
        for i, exp in enumerate(self.experiments):
            exp_info = exp.get('experiment_info', {})
            exp_name = exp_info.get('name', f'Esperimento_{i+1}')
            timestamp = exp.get('timestamp', 'N/A')
            html += f"<div class='experiment'><h3>{exp_name}</h3><p>Data: {timestamp}</p>"
            html += f"<h4>Varianti Prompt</h4><pre>{json.dumps(exp_info.get('prompt_variants', {}), indent=2, ensure_ascii=False)}</pre>"
            
            html += "<h4>Raccomandazioni per Metrica (per Utente)</h4>"
            metric_recs = exp.get('metric_recommendations', {})
            if not metric_recs: html += "<p>N/A</p>"
            for user_id, user_metrics in metric_recs.items():
                html += f"<h5>Utente {user_id}</h5><table><tr><th>Metrica</th><th>Raccomandazioni</th><th>Spiegazione</th></tr>"
                for metric_name, metric_data in user_metrics.items():
                    recs = metric_data.get('recommendations', [])
                    recs_str = ", ".join([get_movie_title(r) for r in recs])
                    expl = metric_data.get('explanation', 'N/A')
                    html += f"<tr><td>{metric_name}</td><td>{recs_str}</td><td>{expl}</td></tr>"
                html += "</table>"
            
            final_eval = exp.get('final_evaluation', {})
            html += "<h4>Valutazione Finale Aggregata</h4><table>"
            final_recs = final_eval.get('final_recommendations', [])
            final_recs_str = ", ".join([get_movie_title(r) for r in final_recs])
            html += f"<tr><th>Raccomandazioni Finali</th><td>{final_recs_str}</td></tr>"
            html += f"<tr><th>Giustificazione</th><td>{final_eval.get('justification', 'N/A')}</td></tr>"
            html += f"<tr><th>Trade-offs</th><td>{final_eval.get('trade_offs', 'N/A')}</td></tr></table>"

            exp_metrics = exp.get('metrics', {})
            if exp_metrics and not exp_metrics.get('error'):
                html += "<h4>Metriche Calcolate</h4><div class='metrics-container'>"
                html += "<div class='metric-block'><h5>Precision@k</h5><table><tr><th>Metrica</th><th>Valore</th></tr>"
                for key, name in [('precision_at_k', 'Prec@k (Prec)'), ('coverage', 'Prec@k (Cov)'), ('final_recommendations', 'Prec@k (Final)')]:
                    score = exp_metrics.get(key, {}).get('precision_score', 0)
                    html += f"<tr><td>{name}</td><td>{score:.4f}</td></tr>"
                html += "</table></div>"
                html += "<div class='metric-block'><h5>Genre Coverage</h5><table><tr><th>Metrica</th><th>Valore</th></tr>"
                for key, name in [('precision_at_k', 'GenreCov (Prec)'), ('coverage', 'GenreCov (Cov)'), ('final_recommendations', 'GenreCov (Final)')]:
                    score = exp_metrics.get(key, {}).get('genre_coverage', 0)
                    html += f"<tr><td>{name}</td><td>{score:.4f}</td></tr>"
                html += "</table></div>"
                html += "<div class='metric-block'><h5>Total Coverage</h5><table><tr><th>Metrica</th><th>Valore</th></tr>"
                total_cov_score = exp_metrics.get('total_coverage', 0)
                html += f"<tr><td>Item Coverage</td><td>{total_cov_score:.4f}</td></tr>"
                html += "</table></div>"
                html += "</div>"
            elif exp_metrics.get('error'):
                 html += f"<h4>Errore nel calcolo metriche</h4><p>{exp_metrics['error']}</p>"
            html += "</div>"
        
        html += "</body></html>"
        try:
            with open(output_file, 'w', encoding='utf-8') as f: f.write(html)
            return f"Report HTML salvato in {output_file}"
        except Exception as e: return f"Errore salvataggio report HTML: {e}"
    
    def generate_markdown_report(self, output_file="experiment_report.md") -> str:
        """Genera un report Markdown dettagliato."""
        if not self.experiments: return "Nessun esperimento disponibile."
        
        movies_df = None
        try:
             movies_path = os.path.join('data', 'processed', 'movies.csv')
             if os.path.exists(movies_path): movies_df = pd.read_csv(movies_path, index_col='movie_id')
        except Exception as e: print(f"Attenzione: impossibile caricare movies.csv: {e}")
        def get_movie_title(movie_id):
            try: 
                mid_int = int(movie_id)
                if movies_df is not None and mid_int in movies_df.index: return movies_df.loc[mid_int, 'title']
            except: pass
            return f"ID:{movie_id}"
        
        diversity_metrics = self.calculate_diversity_metrics()
        md = f"# Report Esperimenti ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n"
        md += "## Statistiche Generali\n"
        md += f"- **Totale Esperimenti**: {diversity_metrics['total_experiments']}\n"
        unique_final = diversity_metrics['unique_final_recommendations']
        md += f"- **Raccomandazioni Finali Uniche**: {unique_final.get('count', 0)}\n\n"
        md += "### Film più Raccomandati (Finali)\n| Film | Frequenza |\n|---|---|\n"
        freq_items = list(diversity_metrics["recommendation_frequency"].items())[:15]
        for movie_id, frequency in freq_items:
            md += f"| {get_movie_title(movie_id)} | {frequency} |\n"
        md += "\n"

        md += "## Performance per Metrica (Aggregata)\n"
        for metric, data in diversity_metrics.get("metric_performance", {}).items():
            md += f"### Metrica: {metric}\n"
            unique_metric = data.get("unique_recommendations", {})
            md += f"- **Raccomandazioni Uniche ({metric})**: {unique_metric.get('count', 0)}\n"
            themes = data.get("explanation_themes", {})
            if themes:
                md += "#### Temi nelle Spiegazioni\n| Tema | Frequenza |\n|---|---|\n"
                sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
            for theme, frequency in sorted_themes:
                md += f"| {theme} | {frequency} |\n"
            md += "\n"
        
        md += "## Dettagli Esperimenti\n"
        for i, exp in enumerate(self.experiments):
            exp_info = exp.get('experiment_info', {})
            exp_name = exp_info.get('name', f'Esperimento_{i+1}')
            timestamp = exp.get('timestamp', 'N/A')
            md += f"### {exp_name}\n*Data: {timestamp}*\n\n"
            md += f"#### Varianti Prompt\n```json\n{json.dumps(exp_info.get('prompt_variants', {}), indent=2, ensure_ascii=False)}\n```\n\n"
            
            md += "#### Raccomandazioni per Metrica (per Utente)\n"
            metric_recs = exp.get('metric_recommendations', {})
            if not metric_recs: md += "*N/A*\n"
            for user_id, user_metrics in metric_recs.items():
                md += f"##### Utente {user_id}\n| Metrica | Raccomandazioni | Spiegazione |\n|---|---|---|\n"
                for metric_name, metric_data in user_metrics.items():
                    recs = metric_data.get('recommendations', [])
                    recs_str = ", ".join([get_movie_title(r) for r in recs])
                    expl = metric_data.get('explanation', 'N/A').replace('\n', ' ')
                    md += f"| {metric_name} | {recs_str} | {expl} |\n"
                md += "\n"
            
            final_eval = exp.get('final_evaluation', {})
            md += "#### Valutazione Finale Aggregata\n"
            final_recs = final_eval.get('final_recommendations', [])
            final_recs_str = ", ".join([get_movie_title(r) for r in final_recs])
            md += f"- **Raccomandazioni Finali**: {final_recs_str}\n"
            md += f"- **Giustificazione**: {final_eval.get('justification', 'N/A')}\n"
            md += f"- **Trade-offs**: {final_eval.get('trade_offs', 'N/A')}\n\n"

            exp_metrics = exp.get('metrics', {})
            if exp_metrics and not exp_metrics.get('error'):
                md += "#### Metriche Calcolate\n"
                md += "| Metrica | Precision@k | Genre Coverage |\n|---|---|---|\n"
                for key, name in [('precision_at_k', 'Precision'), ('coverage', 'Coverage'), ('final_recommendations', 'Final')]:
                    prec = exp_metrics.get(key, {}).get('precision_score', 0)
                    gen_cov = exp_metrics.get(key, {}).get('genre_coverage', 0)
                    md += f"| {name} | {prec:.4f} | {gen_cov:.4f} |\n"
                total_cov = exp_metrics.get('total_coverage', 0)
                md += f"\n- **Total Item Coverage**: {total_cov:.4f}\n"
            elif exp_metrics.get('error'):
                 md += f"*Errore nel calcolo metriche: {exp_metrics['error']}*\n"
            md += "\n---\n"
        
        try:
             with open(output_file, 'w', encoding='utf-8') as f: f.write(md)
             return f"Report Markdown salvato in {output_file}"
        except Exception as e:
             return f"Errore salvataggio report MD: {e}"
    
    def compare_prompt_variants(self):
        """Analizza come diverse varianti di prompt influenzano i risultati."""
        if not self.experiments: return {"error": "Nessun esperimento."}
        prompt_groups = {}
        for exp in self.experiments:
            exp_info = exp.get('experiment_info', {})
            prompt_vars = exp_info.get('prompt_variants', {})
            exp_name = exp_info.get('name', 'Unknown')
            if not prompt_vars: continue
            key = json.dumps(prompt_vars, sort_keys=True)
            if key not in prompt_groups:
                prompt_groups[key] = {"prompt_variants": prompt_vars, "experiments": [], "final_recommendations": []}
            prompt_groups[key]["experiments"].append(exp_name)
            final_recs = exp.get('final_evaluation', {}).get('final_recommendations', [])
            if final_recs:
                 # Aggiungi solo recs hashabili
                 hashable_recs = []
                 for r in final_recs:
                     try: hashable_recs.append(int(r))
                     except (ValueError, TypeError): pass
                 prompt_groups[key]["final_recommendations"].extend(hashable_recs)
        
        comparison_results = []
        for key, group in prompt_groups.items():
            valid_recs = [r for r in group["final_recommendations"] if isinstance(r, int)]
            unique_recs = set(valid_recs)
            comparison_results.append({
                "prompt_variants": group["prompt_variants"],
                "experiments_count": len(group["experiments"]),
                "unique_final_recs_count": len(unique_recs),
                "experiments": sorted(group["experiments"])
            })
        return sorted(comparison_results, key=lambda x: x['experiments_count'], reverse=True)

    def run_comprehensive_analysis(self, output_dir="reports") -> Dict:
        """Esegue un'analisi completa e genera report."""
        if not self.experiments: print("Nessun esperimento caricato per l'analisi."); return {}
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nAvvio analisi completa su {len(self.experiments)} esperimenti...")
        
        report_paths = []
        generated_reports = {}
        
        try: generated_reports["html"] = self.generate_html_report(os.path.join(output_dir, "experiment_report.html"))
        except Exception as e: print(f"Errore report HTML: {e}"); generated_reports["html"] = f"Errore: {e}"
        if "Errore" not in generated_reports["html"]: report_paths.append(generated_reports["html"]) 
        
        try: generated_reports["markdown"] = self.generate_markdown_report(os.path.join(output_dir, "experiment_report.md"))
        except Exception as e: print(f"Errore report MD: {e}"); generated_reports["markdown"] = f"Errore: {e}"
        if "Errore" not in generated_reports["markdown"]: report_paths.append(generated_reports["markdown"])

        diversity_metrics = {}
        try: 
             diversity_metrics = self.calculate_diversity_metrics()
             diversity_path = os.path.join(output_dir, "diversity_metrics.json")
             with open(diversity_path, 'w', encoding='utf-8') as f: json.dump(diversity_metrics, f, ensure_ascii=False, indent=2)
             report_paths.append(diversity_path); generated_reports["diversity"] = diversity_path
        except Exception as e: print(f"Errore metriche diversità: {e}"); generated_reports["diversity"] = f"Errore: {e}"

        prompt_comparison = []
        try:
             prompt_comparison = self.compare_prompt_variants()
             prompt_comp_path = os.path.join(output_dir, "prompt_comparison.json")
             with open(prompt_comp_path, 'w', encoding='utf-8') as f: json.dump(prompt_comparison, f, ensure_ascii=False, indent=2)
             report_paths.append(prompt_comp_path); generated_reports["prompt_comparison"] = prompt_comp_path
        except Exception as e: print(f"Errore comparazione prompt: {e}"); generated_reports["prompt_comparison"] = f"Errore: {e}"
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(self.experiments),
            "unique_final_recommendations": diversity_metrics.get('unique_final_recommendations',{}).get('count', 0),
            "metrics_analyzed": list(diversity_metrics.get('metric_performance', {}).keys()),
            "prompt_variant_groups": len(prompt_comparison),
            "reports_generated_status": generated_reports
        }
        try:
            summary_path = os.path.join(output_dir, "analysis_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f: json.dump(summary, f, ensure_ascii=False, indent=2)
            report_paths.append(summary_path)
        except Exception as e: print(f"Errore salvataggio summary: {e}")
            
        print("\nAnalisi completata.")
        return summary