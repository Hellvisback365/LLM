#!/usr/bin/env python3
"""
Script per creare aggregazioni LLM pulite e salvarle in un nuovo file.
Questo script processa i checkpoint esistenti e crea aggregazioni intelligenti
usando solo l'LLM, senza residui del vecchio codice hardcoded.
"""

import asyncio
import json
import os
import glob
import shutil
from typing import Dict, Any, List, Tuple
import sys
from datetime import datetime

# Aggiungi il percorso src al path Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from recommender.core.recommender import RecommenderSystem


class LLMAggregatorReplacer:
    """Sostituisce l'aggregazione hardcoded con quella LLM per tutti gli utenti."""
    
    def __init__(self):
        self.recommender = None
        self.output_file = "llm_aggregated_recommendations.json"  # NUOVO FILE PULITO
        self.backup_file = None  # Non serve backup per nuovo file
        self.catalog_cache = None  # Cache del catalogo per evitare ricaricamenti
        self.batch_size = 50  # Aumentato per maggiore parallelismo
        self.semaphore = asyncio.Semaphore(50)  # Limita chiamate LLM concorrenti
        
    def create_backup(self):
        """Gestisce il file esistente e il resume."""
        if os.path.exists(self.output_file):
            print(f"✓ File esistente trovato: {self.output_file}")
            print("Modalità RESUME: continuerà da dove si è fermato")
        else:
            print("✓ Creazione nuovo file pulito per aggregazioni LLM")
    
    def initialize_recommender(self):
        """Inizializza il sistema di raccomandazione."""
        print("Inizializzazione sistema di raccomandazione...")
        self.recommender = RecommenderSystem()
        self.recommender.initialize_system()
        
        # Pre-carica il catalogo una sola volta per velocità
        print("Pre-caricamento catalogo ottimizzato...")
        self.catalog_cache = self.recommender.get_optimized_catalog(limit=100)
        print(f"✓ Catalogo caricato con {len(self.catalog_cache)} elementi")
        print("Sistema inizializzato.")
    
    def find_all_checkpoints(self) -> List[str]:
        """Trova tutti i file di checkpoint."""
        checkpoint_files = glob.glob("checkpoint_batch_*.json")
        checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        return checkpoint_files
    
    def load_checkpoint(self, checkpoint_file: str) -> Dict[str, Any]:
        """Carica un singolo checkpoint."""
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"ERROR: Errore nel caricare checkpoint {checkpoint_file}: {e}")
            return {}
    
    def extract_user_recommendations(self, checkpoint_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """Estrae le raccomandazioni per utente da un checkpoint."""
        user_recommendations = {}
        
        all_user_results = checkpoint_data.get('user_results', {})
        
        for user_id_str, user_data in all_user_results.items():
            try:
                user_id = int(user_id_str)
                
                precision_result = user_data.get('precision_at_k', {})
                coverage_result = user_data.get('coverage', {})
                
                if precision_result or coverage_result:
                    user_recommendations[user_id] = {
                        'precision_at_k': precision_result,
                        'coverage': coverage_result
                    }
                    
            except (ValueError, TypeError) as e:
                print(f"WARN: Errore nel processare utente {user_id_str}: {e}")
                continue
        
        return user_recommendations
    
    async def aggregate_user_with_llm(self, user_id: int, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggrega le raccomandazioni per un utente usando LLM con semaforo per controllo concorrenza."""
        async with self.semaphore:  # Limita chiamate concorrenti
            try:
                precision_result = user_data.get('precision_at_k', {})
                coverage_result = user_data.get('coverage', {})
                
                if not precision_result and not coverage_result:
                    return None
                
                # Prepara i dati per l'aggregazione LLM
                all_recommendations = {}
                
                if precision_result:
                    all_recommendations["precision_at_k"] = {
                        f"user_{user_id}": precision_result
                    }
                
                if coverage_result:
                    all_recommendations["coverage"] = {
                        f"user_{user_id}": coverage_result
                    }
                
                # Usa il catalogo pre-caricato invece di ricaricarlo ogni volta
                aggregated_result = await self.recommender.evaluate_final_recommendations(
                    all_recommendations, self.catalog_cache
                )
                
                if aggregated_result:
                    return {
                        "user_id": user_id,
                        "aggregated_recommendations": aggregated_result.get("final_recommendations", []),
                        "justification": aggregated_result.get("justification", ""),
                        "trade_offs": aggregated_result.get("trade_offs", ""),
                        "precision_recommendations": precision_result.get("recommendations", []),
                        "coverage_recommendations": coverage_result.get("recommendations", []),
                        "precision_explanation": precision_result.get("explanation", "Based on user preferences and viewing history."),
                        "coverage_explanation": coverage_result.get("explanation", "A diverse set of movies to maximize exploration."),
                        "aggregation_method": "llm_intelligent",
                        "aggregation_timestamp": datetime.now().isoformat()
                    }
                else:
                    return None
                    
            except Exception as e:
                print(f"ERROR: Errore nell'aggregazione LLM per utente {user_id}: {e}")
                return None
    
    async def process_users_batch(self, users_batch: List[Tuple[int, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Processa un batch di utenti in parallelo per massima velocità."""
        tasks = []
        for user_id, user_data in users_batch:
            task = self.aggregate_user_with_llm(user_id, user_data)
            tasks.append(task)
        
        # Esegui tutte le aggregazioni LLM in parallelo
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtra i risultati validi
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                user_id = users_batch[i][0]
                print(f"ERROR: Batch error per utente {user_id}: {result}")
            elif result is not None:
                valid_results.append(result)
        
        return valid_results
    
    async def process_checkpoint_batch(self, checkpoint_files: List[str], processed_users_set: set, 
                                     output_data: Dict[str, Any], start_index: int) -> Tuple[int, int, int]:
        """Processa un batch di checkpoint in parallelo per massima velocità."""
        total_processed = 0
        total_success = 0
        total_failures = 0
        
        # Carica tutti i checkpoint del batch
        all_users_to_process = []
        
        for checkpoint_file in checkpoint_files:
            checkpoint_data = self.load_checkpoint(checkpoint_file)
            if not checkpoint_data:
                continue
                
            user_recommendations = self.extract_user_recommendations(checkpoint_data)
            users_to_process = {uid: data for uid, data in user_recommendations.items() 
                              if uid not in processed_users_set}
            
            # Aggiungi tutti gli utenti alla lista generale
            for user_id, user_data in users_to_process.items():
                all_users_to_process.append((user_id, user_data))
        
        if not all_users_to_process:
            return 0, 0, 0
        
        # Processa tutto in mega-batch per massima velocità
        print(f"📦 MEGA-BATCH: Processando {len(all_users_to_process)} utenti da {len(checkpoint_files)} checkpoint in parallelo...")
        
        # Suddividi in chunk per evitare overflow di memoria
        chunk_size = 100  # Processa 100 utenti alla volta
        for chunk_start in range(0, len(all_users_to_process), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(all_users_to_process))
            current_chunk = all_users_to_process[chunk_start:chunk_end]
            
            # Processa il chunk in parallelo
            chunk_results = await self.process_users_batch(current_chunk)
            
            # Salva i risultati
            chunk_success = 0
            for result in chunk_results:
                if result:
                    user_id_str = str(result["user_id"])
                    output_data["user_aggregated_recommendations"][user_id_str] = result
                    chunk_success += 1
                    processed_users_set.add(result["user_id"])  # Aggiorna il set
            
            total_success += chunk_success
            total_failures += (len(current_chunk) - chunk_success)
            total_processed += len(current_chunk)
            
            print(f"🔥 Chunk {chunk_start//chunk_size + 1}: {chunk_success}/{len(current_chunk)} successi")
        
        return total_processed, total_success, total_failures
    
    async def process_checkpoint_batch_fallback(self, checkpoint_files: List[str], 
                                              processed_users_set: set, output_data: Dict[str, Any]):
        """Fallback method: processa checkpoint uno alla volta in caso di errore nel super-batch."""
        print("🔄 Fallback: processamento sequenziale per questo batch")
        for checkpoint_file in checkpoint_files:
            checkpoint_data = self.load_checkpoint(checkpoint_file)
            if not checkpoint_data:
                continue
                
            user_recommendations = self.extract_user_recommendations(checkpoint_data)
            users_to_process = {uid: data for uid, data in user_recommendations.items() 
                              if uid not in processed_users_set}
            
            if users_to_process:
                users_list = list(users_to_process.items())
                results = await self.process_users_batch(users_list)
                
                for result in results:
                    if result:
                        user_id_str = str(result["user_id"])
                        output_data["user_aggregated_recommendations"][user_id_str] = result
                        processed_users_set.add(result["user_id"])
    
    def load_existing_output(self) -> Dict[str, Any]:
        """Carica il file di output esistente per il resume."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                existing_users = len(data.get("user_aggregated_recommendations", {}))
                print(f"✓ Caricato file esistente con {existing_users} utenti già processati")
                return data
            except Exception as e:
                print(f"WARN: Errore nel caricare file esistente: {e}")
                return {"user_aggregated_recommendations": {}}
        else:
            return {"user_aggregated_recommendations": {}}
    
    def get_processed_users(self, output_data: Dict[str, Any]) -> set:
        """Ottiene la lista degli utenti già processati."""
        processed = set()
        for user_id_str in output_data.get("user_aggregated_recommendations", {}):
            try:
                processed.add(int(user_id_str))
            except (ValueError, TypeError):
                continue
        return processed
    
    async def process_all_checkpoints(self, max_users: int = None):
        """Processa tutti i checkpoint sostituendo con aggregazione LLM."""
        print("=== SOSTITUZIONE AGGREGAZIONE HARDCODED -> LLM ===")
        
        # Crea backup del file esistente
        self.create_backup()
        
        # Trova tutti i checkpoint
        checkpoint_files = self.find_all_checkpoints()
        print(f"Trovati {len(checkpoint_files)} checkpoint da processare")
        
        if not checkpoint_files:
            print("ERROR: Nessun checkpoint trovato!")
            return
        
        # Inizializza il sistema di raccomandazione
        self.initialize_recommender()
        
        # Inizializza la struttura del file di output
        output_data = self.load_existing_output()
        processed_users = self.get_processed_users(output_data)
        
        print(f"Utenti già processati: {len(processed_users)}")
        if len(processed_users) > 0:
            print(f"Resume dal checkpoint: utenti {min(processed_users)}-{max(processed_users)} già completati")
        
        # Ottimizzazione: converti in set per lookup O(1)
        processed_users_set = set(processed_users)
        
        # Contatori per il progresso
        total_users_processed = len(processed_users)  # Inizia dal numero già processato
        total_llm_success = len(processed_users)      # Utenti già processati sono successi
        total_llm_failures = 0
        
        start_time = datetime.now()
        
        # SUPER OTTIMIZZAZIONE: Processa checkpoint in gruppi per mega-batch
        checkpoint_batch_size = 10  # Processa 10 checkpoint alla volta
        
        for batch_start in range(0, len(checkpoint_files), checkpoint_batch_size):
            if max_users and total_users_processed >= max_users:
                break
                
            batch_end = min(batch_start + checkpoint_batch_size, len(checkpoint_files))
            checkpoint_batch = checkpoint_files[batch_start:batch_end]
            
            print(f"\n🚀 SUPER-BATCH {batch_start//checkpoint_batch_size + 1}: Processando checkpoint {batch_start+1}-{batch_end}")
            
            # Disabilita output per velocità estrema
            if total_users_processed > 5:
                devnull_file = open(os.devnull, 'w')
                original_stdout = sys.stdout
                sys.stdout = devnull_file
            
            try:
                # Processa l'intero batch di checkpoint in parallelo
                batch_processed, batch_success, batch_failures = await self.process_checkpoint_batch(
                    checkpoint_batch, processed_users_set, output_data, batch_start
                )
                
                # Riabilita output
                if total_users_processed > 5:
                    sys.stdout = original_stdout
                    devnull_file.close()
                
                total_users_processed += batch_processed
                total_llm_success += batch_success
                total_llm_failures += batch_failures
                
                if batch_processed > 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    avg_time = elapsed / total_users_processed if total_users_processed > 0 else 0
                    remaining = (6040 - total_users_processed) * avg_time
                    
                    print(f"⚡ SUPER-PROGRESS: {total_users_processed}/6040 utenti")
                    print(f"🎯 LLM success: {total_llm_success}/{total_users_processed} ({total_llm_success/total_users_processed*100:.1f}%)")
                    print(f"⏱️  Tempo medio: {avg_time:.3f}s/utente")
                    print(f"🕐 Tempo rimanente: {remaining/60:.1f} minuti")
                    print(f"📦 Batch size: {checkpoint_batch_size} checkpoint, {self.batch_size} utenti paralleli")
                    
                    # Salva checkpoint ogni super-batch
                    try:
                        with open(self.output_file, 'w', encoding='utf-8') as f:
                            json.dump(output_data, f, separators=(',', ':'), ensure_ascii=False)
                        print("💾 Super-checkpoint salvato")
                    except Exception as e:
                        print(f"WARN: Errore nel salvare super-checkpoint: {e}")
                else:
                    print("✓ Nessun nuovo utente trovato in questo batch. Utenti già processati in precedenza.")
                
            except Exception as e:
                # Riabilita output in caso di errore
                if total_users_processed > 5:
                    sys.stdout = original_stdout
                    if 'devnull_file' in locals():
                        devnull_file.close()
                
                print(f"ERROR: Errore nel super-batch: {e}")
                # Fallback al metodo originale per questo batch
                await self.process_checkpoint_batch_fallback(checkpoint_batch, processed_users_set, output_data)
        
        # Salvataggio finale
        try:
            print(f"\nSalvataggio finale di {total_llm_success} aggregazioni LLM...")
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)  # Solo finale con indent
            print("✓ File finale salvato")
        except Exception as e:
            print(f"ERROR: Errore nel salvare file finale: {e}")
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n=== SOSTITUZIONE COMPLETATA ===")
        print(f"Utenti totali processati: {total_users_processed}")
        print(f"LLM aggregazioni riuscite: {total_llm_success}")
        print(f"LLM aggregazioni fallite: {total_llm_failures}")
        print(f"Tasso di successo LLM: {total_llm_success/total_users_processed*100:.1f}%")
        print(f"Tempo totale: {total_time/60:.1f} minuti")
        print(f"Tempo medio per utente: {total_time/total_users_processed:.2f} secondi")
        print(f"File di output: {self.output_file}")
        print("✅ AGGREGAZIONI LLM COMPLETATE!")


async def main():
    """Funzione principale."""
    
    print("🤖 CREAZIONE AGGREGAZIONI LLM PULITE - VERSIONE SUPER-OTTIMIZZATA")
    print("Questo script creerà aggregazioni LLM intelligenti in un nuovo file pulito.")
    print("Le aggregazioni saranno generate solo dall'LLM, senza residui del vecchio codice.")
    print("SUPER-OTTIMIZZAZIONI ATTIVE:")
    print("- Processamento mega-batch (10 checkpoint + 50 utenti in parallelo)")
    print("- Semaforo per controllo concorrenza LLM (50 chiamate simultanee)")
    print("- Cache del catalogo (riuso invece di ricaricamento)")
    print("- Super-checkpoint intermedi per gruppi di checkpoint")
    print("- Output debug completamente disabilitato")
    print("File di output: llm_aggregated_recommendations.json")
    
    # Verifica se esiste già un file per il resume
    if os.path.exists("llm_aggregated_recommendations.json"):
        try:
            with open("llm_aggregated_recommendations.json", 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            existing_users = len(existing_data.get("user_aggregated_recommendations", {}))
            if existing_users > 0:
                print(f"\n🔄 MODALITÀ RESUME ATTIVATA!")
                print(f"Trovato file esistente con {existing_users} utenti già processati.")
                print("Lo script continuerà dal punto dove si era fermato.")
        except:
            pass
    
    # Procedi automaticamente con tutti gli utenti (modalità resume)
    print("\n🚀 MODALITÀ AUTOMATICA: Procedendo con TUTTI i 6040 utenti")
    print("Per interrompere usa Ctrl+C")
    
    replacer = LLMAggregatorReplacer()
    await replacer.process_all_checkpoints()


if __name__ == "__main__":
    asyncio.run(main())
